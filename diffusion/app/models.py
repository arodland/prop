import torch
import math
import diffusers
from datasets import load_dataset, Image
import torchvision.transforms.v2 as transforms
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
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
            sample_size=(184, 368),
            block_out_channels=(128, 128, 256, 256),
            dropout=0.1,
            down_block_types=(
                "DownBlock2D",       # a regular ResNet downsampling block
                "DownBlock2D",
                "AttnDownBlock2D",   # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",     # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",         # a regular ResNet upsampling block
            ),
        )
        self.model.to(memory_format=torch.channels_last)
        self.model = torch.compile(self.model, mode="max-autotune")

        # self.scheduler = diffusers.schedulers.DDIMScheduler(
        #     # beta_schedule="squaredcos_cap_v2",
        #     rescale_betas_zero_snr=True,
        # )
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            thresholding=True,
            rescale_betas_zero_snr=True,
        )

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = modified_mse_loss(residual, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch["images"]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = modified_mse_loss(residual, noise)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        noise = torch.randn_like(images)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        loss = modified_mse_loss(residual, noise)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        pipe = diffusers.DDPMPipeline(self.model, self.scheduler)
        pipe = pipe.to(device=self.device)
        with torch.inference_mode():
            (pil_images, ) = pipe(
                batch_size=9,
                num_inference_steps=20,
                output_type="pil",
                return_dict=False
            )
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
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 46 * 92, 5),
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

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
        toys = [ (dt - dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)) /
                 datetime.timedelta(days=365) for dt in dts ]
        tods = [ (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)) /
                 datetime.timedelta(hours=24) for dt in dts ]
        sin_toy = torch.sin(torch.tensor(toys) * 2 * math.pi)
        cos_toy = torch.cos(torch.tensor(toys) * 2 * math.pi)
        sin_tod = torch.sin(torch.tensor(tods) * 2 * math.pi)
        cos_tod = torch.cos(torch.tensor(tods) * 2 * math.pi)
        targets = torch.stack([sin_toy, cos_toy, sin_tod, cos_tod, torch.tensor(sample["ssn"]) / 100. - 1.], dim=1)
        return {"images": images, "target": targets}

    def prepare_data(self):
        load_dataset("arodland/IRI-iono-maps", self.metric)

    def setup(self, stage=None):
        dataset = load_dataset("arodland/IRI-iono-maps", self.metric)["train"]
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
