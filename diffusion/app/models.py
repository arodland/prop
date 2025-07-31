import torch
import math
import diffusers
from datasets import load_dataset, Image
import torchvision.transforms.v2 as transforms
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from diffusers.utils import numpy_to_pil
import datetime

torch.set_float32_matmul_precision("high")
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True

# MSE loss, modified to be approximately the loss of the worst channel,
# but differentiable.
def modified_mse_loss(prediction, target):
    squared_errors = torch.pow(prediction - target, 2)
    channel_sums = torch.mean(squared_errors, dim=(2, 3))  # Average over height and width
    # Log-sum-exp over channels to weight towards the worst channel
    channel_weighted = torch.logsumexp(channel_sums, dim=1) - math.log(squared_errors.shape[1])
    return torch.mean(channel_weighted) # Average over the batch

class DiffusionModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = diffusers.models.UNet2DModel(
            sample_size=(24, 48),
            in_channels=4,
            out_channels=4,
            block_out_channels=(64, 128, 256, 512),
            dropout=0.1,
            down_block_types=(
                "DownBlock2D",       # a regular ResNet downsampling block
                "AttnDownBlock2D",
                "AttnDownBlock2D",   # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",     # a ResNet upsampling block with spatial self-attention
                "AttnUpBlock2D",
                "UpBlock2D",         # a regular ResNet upsampling block
            ),
        )
        self.model.to(memory_format=torch.channels_last)
        self.model = torch.compile(self.model, mode="max-autotune")

        self.vae = diffusers.models.AutoencoderTiny.from_pretrained("./taesd-iono-finetuned")

        # self.scheduler = diffusers.schedulers.DDIMScheduler(
        #     # beta_schedule="squaredcos_cap_v2",
        #     rescale_betas_zero_snr=True,
        # )
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            thresholding=False,
            rescale_betas_zero_snr=False,
        )
        self.inference_scheduler = diffusers.schedulers.DDPMScheduler(
            thresholding=False,
            rescale_betas_zero_snr=False,
        )

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            images = batch["images"]
            # print("images:", summarize_tensor(images))
            # tv.utils.save_image(images[0], "out/in_images.png")
            latents_raw = self.vae.encode(images * 2.0 - 1.0).latents
            # print("latents_raw:", summarize_tensor(latents_raw))
            latents = F.pad(latents_raw / self.vae.latent_magnitude, (0, 2, 0, 1))
            # tv.utils.save_image(latents[0] * 2.0 + 1.0, "out/in_latents.png")
            # print("latents:", summarize_tensor(latents))
        decoded = (self.vae.decode(latents_raw).sample + 1.0) / 2.0  # Scale to [0, 1]
        # print("decoded:", summarize_tensor(decoded))
        # tv.utils.save_image(decoded[0], "out/in_decoded.png")
        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)
        residual = self.model(noisy_latents, steps).sample
        loss = F.mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            images = batch["images"]
            latents_raw = self.vae.encode(images * 2.0 - 1.0).latents
            latents = F.pad(latents_raw / self.vae.latent_magnitude, (0, 2, 0, 1))
        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)
        residual = self.model(noisy_latents, steps).sample
        loss = F.mse_loss(residual, noise)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            images = batch["images"]
            latents_raw = self.vae.encode(images * 2.0 - 1.0).latents
            latents = F.pad(latents_raw / self.vae.latent_magnitude, (0, 2, 0, 1))
        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)
        residual = self.model(noisy_latents, steps).sample
        loss = F.mse_loss(residual, noise)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        # gamma=0.93 will lose about one order of magnitude in 30 epochs
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.93)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            eta_min=1e-6,
        )
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        with torch.no_grad():
            x = torch.randn(9, 4, 24, 48, device=self.device)
            self.inference_scheduler.set_timesteps(20, device=self.device)
            for t in self.inference_scheduler.timesteps:
                model_output = self.model(x, t).sample
                step = self.inference_scheduler.step(model_output, t, x)
                x = step.prev_sample

            latents = step.pred_original_sample[..., :23, :46]
            pil_latents = numpy_to_pil(((latents[:, :3, ...] + 1.0) / 2.0).cpu().permute(0, 2, 3, 1).detach().numpy())
            latents = latents * self.vae.latent_magnitude
            tv.utils.save_image(
                tv.utils.make_grid(
                    torch.stack([tv.transforms.functional.to_tensor(pil_image) for pil_image in pil_latents]),
                    nrow=3,
                ),
                "out/latents.png",
            ),
            image = self.vae.decode(latents).sample
            image = (image + 1.0) / 2.0  # Scale to [0, 1]
            image = image.clip(0.0, 1.0)

            pil_images = numpy_to_pil(image.cpu().permute(0, 2, 3, 1).detach().numpy())

            images = torch.stack([tv.transforms.functional.to_tensor(pil_image.crop((0, 0, 361, 181)))
                                  for pil_image in pil_images])
            image_grid = tv.utils.make_grid(images, nrow=3)

            filename = "out/checkpoint.png"
            tv.utils.save_image(image_grid, filename)
            print(f"Generated images saved to {filename}")

def guidance_loss(prediction, target):
    return F.mse_loss(prediction, target)
    # weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0], device=prediction.device)
    # return F.mse_loss(prediction * weight, target * weight)

class GuidanceModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3),
            nn.Flatten(),
            nn.Linear(128 * 40 * 20, 1024),
            nn.ReLU(),
            nn.Linear(1024, 36),
            nn.Tanh(),
            nn.Linear(36, 6)
        )
        self.model = torch.compile(self.model, mode="max-autotune")

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        pred = self.model(images)
        target = batch["target"]
        loss = guidance_loss(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch["images"]
        pred = self.model(images)
        target = batch["target"]
        loss = guidance_loss(pred, target)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        pred = self.model(images)
        target = batch["target"]
        loss = guidance_loss(pred, target)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=5e-5, decoupled_weight_decay=True)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50,
            eta_min=1e-5,
        )
        return [optimizer], [scheduler]

class VAEModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.vae = diffusers.models.AutoencoderTiny.from_pretrained("./taesd-iono-50k")
        self.vae = torch.compile(self.vae, mode="max-autotune")

    def training_step(self, batch, batch_idx):
        images = batch["images"] * 2.0 - 1.0  # Scale to [-1, 1]
        latents = self.vae.encode(images).latents
        decoded = self.vae.decode(latents).sample

        img_loss = F.mse_loss(decoded, images)
        loss = img_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch["images"] * 2.0 - 1.0  # Scale to [-1, 1]
        latents = self.vae.encode(images).latents
        decoded = self.vae.decode(latents).sample

        img_loss = F.mse_loss(decoded, images)
        loss = img_loss
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"] * 2.0 - 1.0  # Scale to [-1, 1]
        latents = self.vae.encode(images).latents
        decoded = self.vae.decode(latents).sample

        img_loss = F.mse_loss(decoded, images)
        loss = img_loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def forward(self, images):
        return self.vae.encode(images * 2.0 - 1.0).latents

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            eta_min=1e-6,
        )
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        self.vae.save_pretrained("checkpoints/vae_model")

# class GuidedModel(L.LightningModule):
#     def __init__(self, diffusion_model, guidance_model):
#         super().__init__()
#         self.diffusion_model = diffusion_model
#         self.guidance_model = guidance_model
#
#     def forward(self, batch):
#         images = batch["images"]
#         noise = torch.randn_like(images)
#         steps = torch.randint(self.diffusion_model.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
#         noisy_images = self.diffusion_model.scheduler.add_noise(images, noise, steps)
#         residual = self.diffusion_model.model(noisy_images, steps).sample
#         pred = self.guidance_model.model(images)
#         return residual + pred
#

class IRIData(L.LightningDataModule):
    def __init__(self, metric="combined", train_batch=8, test_batch=8, val_batch=8, add_noise=0.0):
        super().__init__()
        self.metric = metric
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.val_batch = val_batch
        self.augment_test = transforms.Compose([
            transforms.ToImage(),
            transforms.RGB(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Pad(padding=(0, 0, 7, 3), fill=0, padding_mode='constant'), # from 361x181 to 368x184
        ])
        if add_noise > 0.0:
            self.augment_train = transforms.Compose([
                transforms.ToImage(),
                transforms.RGB(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.RandomChoice([
                    transforms.Pad(0), # no-op
                    transforms.GaussianNoise(sigma=0.25 * add_noise),
                    transforms.GaussianNoise(sigma=0.5 * add_noise),
                    transforms.GaussianNoise(sigma=0.75 * add_noise),
                    transforms.GaussianNoise(sigma=add_noise),
                ]),
                transforms.Pad(padding=(0, 0, 7, 3), fill=0, padding_mode='constant'), # from 361x181 to 368x184
            ])
        else:
            # Don't waste CPU adding a gaussian blur of 0.
            self.augment_train = self.augment_test

    def dataset_row(self, sample, augment, which):
        images = None
        images = [ augment(image) for image in sample["image"]]

        dts = [ datetime.datetime.fromisoformat(dt) for dt in sample["datetime"] ]
        toys = torch.tensor([ (dt - dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)) /
                              datetime.timedelta(days=365) for dt in dts ])
        tods = torch.tensor([ (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)) /
                              datetime.timedelta(hours=24) for dt in dts ])
        sin_toy = torch.sin(toys * 2 * math.pi)
        cos_toy = torch.cos(toys * 2 * math.pi)
        sin_tod = torch.sin(tods * 2 * math.pi)
        cos_tod = torch.cos(tods * 2 * math.pi)
        years = torch.tensor([ dt.year for dt in dts ], dtype=torch.float32)
        # -1 to 1 is 1950-2050, so the training data covers -0.84 to +0.50
        secular = (years - 2000.0) / 50.

        targets = torch.stack([secular, sin_toy, cos_toy, sin_tod, cos_tod,
                               torch.tensor(sample["ssn"]) / 100. - 1.], dim=1)
        return {"images": images, "target": targets}

    def prepare_data(self):
        load_dataset("arodland/IRI-iono-maps", self.metric)

    def setup(self, stage=None):
        dataset = load_dataset("arodland/IRI-iono-maps", self.metric)["train"]
        # dataset = dataset.filter(lambda sample: sample["datetime"].startswith("2022-") or sample["datetime"].startswith("2023-") or sample["datetime"].startswith("2024-"))
        self.train_dataset, test_dataset = dataset.train_test_split(test_size=0.2, seed=42).values()
        self.test_dataset, self.val_dataset = test_dataset.train_test_split(test_size=0.5, seed=42).values()

        self.train_dataset.set_transform(lambda sample: self.dataset_row(sample, self.augment_train, "train"))
        self.test_dataset.set_transform(lambda sample: self.dataset_row(sample, self.augment_test, "test"))
        self.val_dataset.set_transform(lambda sample: self.dataset_row(sample, self.augment_test, "val"))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_batch, shuffle=True, num_workers=16)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.test_batch, shuffle=False, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.val_batch, shuffle=False, num_workers=4)
