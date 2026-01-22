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
from util import scale_to_diffusion, scale_from_diffusion
import datetime
from util import summarize_tensor

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
            latents_raw = self.vae.encode(scale_to_diffusion(images)).latents
            # print("latents_raw:", summarize_tensor(latents_raw))
            latents = F.pad(self.unscale_latents(latents_raw), (0, 2, 0, 1))
            # tv.utils.save_image(latents[0] * 2.0 + 1.0, "out/in_latents.png")
            # print("latents:", summarize_tensor(latents))
        decoded = scale_from_diffusion(self.vae.decode(latents_raw).sample)  # Scale to [0, 1]
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
            latents_raw = self.vae.encode(scale_to_diffusion(images)).latents
            latents = F.pad(self.unscale_latents(latents_raw), (0, 2, 0, 1))
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
            latents_raw = self.vae.encode(scale_to_diffusion(images)).latents
            latents = F.pad(self.unscale_latents(latents_raw), (0, 2, 0, 1))
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
            T_0=25,
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
            pil_latents = numpy_to_pil(scale_from_diffusion(
                latents[:, :3, ...]).cpu().permute(0, 2, 3, 1).detach().numpy())
            latents = self.scale_latents(latents)
            tv.utils.save_image(
                tv.utils.make_grid(
                    torch.stack([tv.transforms.functional.to_tensor(pil_image) for pil_image in pil_latents]),
                    nrow=3,
                ),
                "out/latents.png",
            ),
            image = self.vae.decode(latents).sample
            image = scale_from_diffusion(image)  # Scale to [0, 1]
            image = image.clip(0.0, 1.0)

            pil_images = numpy_to_pil(image.cpu().permute(0, 2, 3, 1).detach().numpy())

            images = torch.stack([tv.transforms.functional.to_tensor(pil_image.crop((0, 0, 361, 181)))
                                  for pil_image in pil_images])
            image_grid = tv.utils.make_grid(images, nrow=3)

            filename = "out/checkpoint.png"
            tv.utils.save_image(image_grid, filename)
            print(f"Generated images saved to {filename}")

    def scale_latents(self, latents):
        """Scale latents by vae.latent_magnitude for decoding."""
        return latents * self.vae.latent_magnitude

    def unscale_latents(self, latents):
        """Unscale latents by vae.latent_magnitude after encoding."""
        return latents / self.vae.latent_magnitude

class RopeEncoder(L.LightningModule):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        freq = (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freq", freq)

        self.scrambler = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        nn.init.eye_(self.scrambler[0].weight)
        nn.init.zeros_(self.scrambler[0].bias)
        nn.init.eye_(self.scrambler[3].weight)
        nn.init.zeros_(self.scrambler[3].bias)

    def rope_enc(self, x):
        positions = x.type_as(self.freq)
        freqs = torch.einsum("bi, j -> bij", positions, self.freq) # (batch_size, seq_len, dim/2)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb.flatten(1)

    def forward(self, x):
        return self.scrambler(self.rope_enc(x))

# DiT (Diffusion Transformer) Helper Modules

class TimestepEmbedder(nn.Module):
    """
    Embeds timesteps using sinusoidal encoding followed by MLP.
    Standard timestep embedding from the DiT paper.
    """
    def __init__(self, hidden_size=768, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ContinuousClassEmbedder(nn.Module):
    """
    Projects continuous embeddings (from RopeEncoder) to hidden_size.
    Unlike standard DiT which uses discrete class indices, this handles
    continuous 256-dim conditioning vectors.
    """
    def __init__(self, input_size=256, hidden_size=768):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        Args:
            x: (batch, input_size) continuous embeddings
        Returns:
            (batch, hidden_size) projected embeddings
        """
        return self.norm(self.linear(x))


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm (AdaLN-Zero) conditioning.
    Combines self-attention and feedforward with modulation from conditioning.
    """
    def __init__(self, hidden_size=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

        # AdaLN modulation: produces scale, shift, and gate parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Initialize modulation to zero for stable training (identity at initialization)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        """
        Args:
            x: (batch, seq_len, hidden_size) input tokens
            c: (batch, hidden_size) conditioning vector
        Returns:
            (batch, seq_len, hidden_size) output tokens
        """
        # Generate modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-attention block with modulation
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # MLP block with modulation
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class FinalLayer(nn.Module):
    """
    Final layer of DiT: adaptive norm + linear projection + unpatchify.
    """
    def __init__(self, hidden_size=768, patch_size=2, out_channels=4):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        # Initialize to zero for stable training
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.patch_size = patch_size
        self.out_channels = out_channels

    def unpatchify(self, x, h, w):
        """
        Args:
            x: (batch, num_patches, patch_size**2 * channels)
            h, w: height and width in patches
        Returns:
            (batch, channels, H, W) where H=h*patch_size, W=w*patch_size
        """
        c = self.out_channels
        p = self.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return x

    def forward(self, x, c, h, w):
        """
        Args:
            x: (batch, num_patches, hidden_size) input tokens
            c: (batch, hidden_size) conditioning vector
            h, w: spatial dimensions in patches
        Returns:
            (batch, out_channels, H, W) output image
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        x = self.unpatchify(x, h, w)
        return x


class ConditionedDiffusionModel(L.LightningModule):
    def __init__(self,
                 pred_type='epsilon'):
        super().__init__()
        self.save_hyperparameters()
        self.param_encoder = RopeEncoder(dim=64)

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
            class_embed_type="identity",
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
            prediction_type=self.hparams.pred_type,
        )
        self.inference_scheduler = diffusers.schedulers.DDPMScheduler(
            thresholding=False,
            rescale_betas_zero_snr=False,
            prediction_type=self.hparams.pred_type,
        )

    def model_loss(self, prediction, x, noise, step):
        if self.hparams.pred_type == 'epsilon':
            target = noise
        elif self.hparams.pred_type == 'v_prediction':
            alpha_t = self.scheduler.alphas_cumprod[step].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            sigma_t = (1 - self.scheduler.alphas_cumprod[step]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            target = alpha_t.sqrt() * noise - sigma_t.sqrt() * x
        else:
            raise ValueError(f"Unknown prediction type: {self.hparams.pred_type}")
        return F.mse_loss(prediction, target)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            images = batch["images"]
            # print("images:", summarize_tensor(images))
            # tv.utils.save_image(images[0], "out/in_images.png")
            latents_raw = self.vae.encode(scale_to_diffusion(images)).latents
            # print("latents_raw:", summarize_tensor(latents_raw))
            latents = F.pad(self.unscale_latents(latents_raw), (0, 2, 0, 1))
            # tv.utils.save_image(latents[0] * 2.0 + 1.0, "out/in_latents.png")
            # print("latents:", summarize_tensor(latents))
        encoded_targets = self.param_encoder(batch["raw_target"])

        # Dropout targets 20% of the time to allow CFG.
        use_targets = torch.rand(encoded_targets.size(0), device=encoded_targets.device) < 0.8
        encoded_targets = encoded_targets * use_targets.unsqueeze(1).float()

        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)
        model_pred = self.model(noisy_latents, steps, class_labels=encoded_targets).sample
        loss = self.model_loss(model_pred, latents, noise, steps)
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            images = batch["images"]
            latents_raw = self.vae.encode(scale_to_diffusion(images)).latents
            latents = F.pad(self.unscale_latents(latents_raw), (0, 2, 0, 1))
        encoded_targets = self.param_encoder(batch["raw_target"])
        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)
        model_pred = self.model(noisy_latents, steps, class_labels=encoded_targets).sample
        loss = self.model_loss(model_pred, latents, noise, steps)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            images = batch["images"]
            latents_raw = self.vae.encode(scale_to_diffusion(images)).latents
            latents = F.pad(self.unscale_latents(latents_raw), (0, 2, 0, 1))
        encoded_targets = self.param_encoder(batch["raw_target"])
        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)
        model_pred = self.model(noisy_latents, steps, class_labels=encoded_targets).sample
        loss = self.model_loss(model_pred, latents, noise, steps)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        # gamma=0.93 will lose about one order of magnitude in 30 epochs
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.93)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=25,
            eta_min=1e-6,
        )
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        with torch.no_grad():
            x = torch.randn(9, 4, 24, 48, device=self.device)
            encoded_targets = torch.zeros(9, 256, device=self.device)

            self.inference_scheduler.set_timesteps(20, device=self.device)
            for t in self.inference_scheduler.timesteps:
                model_output = self.model(x, t, class_labels=encoded_targets).sample
                step = self.inference_scheduler.step(model_output, t, x)
                x = step.prev_sample

            latents = step.pred_original_sample[..., :23, :46]
            pil_latents = numpy_to_pil(scale_from_diffusion(
                latents[:, :3, ...]).cpu().permute(0, 2, 3, 1).detach().numpy())
            latents = self.scale_latents(latents)
            tv.utils.save_image(
                tv.utils.make_grid(
                    torch.stack([tv.transforms.functional.to_tensor(pil_image) for pil_image in pil_latents]),
                    nrow=3,
                ),
                "out/latents.png",
            ),
            image = self.vae.decode(latents).sample
            image = scale_from_diffusion(image)  # Scale to [0, 1]
            image = image.clip(0.0, 1.0)

            pil_images = numpy_to_pil(image.cpu().permute(0, 2, 3, 1).detach().numpy())

            images = torch.stack([tv.transforms.functional.to_tensor(pil_image.crop((0, 0, 361, 181)))
                                  for pil_image in pil_images])
            image_grid = tv.utils.make_grid(images, nrow=3)

            filename = "out/checkpoint.png"
            tv.utils.save_image(image_grid, filename)
            print(f"Generated images saved to {filename}")

    def scale_latents(self, latents):
        """Scale latents by vae.latent_magnitude for decoding."""
        return latents * self.vae.latent_magnitude

    def unscale_latents(self, latents):
        """Unscale latents by vae.latent_magnitude after encoding."""
        return latents / self.vae.latent_magnitude


class CustomDiTBackbone(nn.Module):
    """
    DiT-B/2 backbone with continuous class conditioning.

    Architecture:
    - Input: (batch, 4, 24, 48) latents
    - Patchify: 2x2 patches → 288 tokens (12×24)
    - 12 transformer blocks with AdaLN-Zero modulation
    - Output: (batch, 4, 24, 48) denoised latents

    Conditioning:
    - Timestep: sinusoidal + MLP → 768-dim
    - Class: continuous 256-dim → 768-dim via ContinuousClassEmbedder
    - Combined additively to modulate each transformer block
    """
    def __init__(
        self,
        input_size=(24, 48),
        patch_size=2,
        in_channels=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        class_embed_size=256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # Calculate number of patches
        self.input_h, self.input_w = input_size
        self.num_patches_h = self.input_h // patch_size
        self.num_patches_w = self.input_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patchify: conv with kernel=stride=patch_size
        self.x_embedder = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size, bias=True
        )

        # Positional embeddings for patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # Continuous class embedder (for RopeEncoder output)
        self.y_embedder = ContinuousClassEmbedder(class_embed_size, hidden_size)

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

        # Initialize
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch embedding like nn.Linear
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, timesteps, class_labels=None):
        """
        Args:
            x: (batch, 4, 24, 48) noisy latents
            timesteps: (batch,) timestep indices
            class_labels: (batch, 256) continuous class embeddings (optional)
        Returns:
            (batch, 4, 24, 48) predicted noise/velocity
        """
        batch_size = x.shape[0]

        # Patchify: (B, 4, 24, 48) → (B, 768, 12, 24)
        x = self.x_embedder(x)
        # Flatten: (B, 768, 12, 24) → (B, 288, 768)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embeddings
        x = x + self.pos_embed

        # Get timestep embeddings
        t = self.t_embedder(timesteps)  # (B, 768)

        # Get class embeddings
        if class_labels is not None:
            y = self.y_embedder(class_labels)  # (B, 768)
        else:
            y = torch.zeros_like(t)

        # Combine conditioning: timestep + class (additive)
        c = t + y  # (B, 768)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer: (B, 288, 768) → (B, 4, 24, 48)
        x = self.final_layer(x, c, self.num_patches_h, self.num_patches_w)

        return x


class ModelOutputWrapper(nn.Module):
    """
    Wraps a model to return output with .sample attribute.
    Ensures interface compatibility with diffusers UNet models.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, timesteps, class_labels=None):
        output = self.model(x, timesteps, class_labels=class_labels)

        # Create a simple object with .sample attribute
        class Output:
            def __init__(self, sample):
                self.sample = sample

        return Output(output)


class DiTDiffusionModel(L.LightningModule):
    """
    DiT-based diffusion model (drop-in replacement for ConditionedDiffusionModel).

    Uses Diffusion Transformer (DiT-B/2) instead of UNet backbone.
    All other components (RopeEncoder, VAE, scheduler) remain identical.
    """
    def __init__(self, pred_type='epsilon'):
        super().__init__()
        self.save_hyperparameters()

        # Same param encoder as ConditionedDiffusionModel
        self.param_encoder = RopeEncoder(dim=64)

        # DiT-B/2 backbone instead of UNet
        dit_backbone = CustomDiTBackbone(
            input_size=(24, 48),
            patch_size=2,
            in_channels=4,
            hidden_size=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            class_embed_size=256,
        )

        # Wrap with output compatibility layer and compile
        self.model = ModelOutputWrapper(dit_backbone)
        self.model = torch.compile(self.model, mode="max-autotune")

        # Same VAE as ConditionedDiffusionModel
        self.vae = diffusers.models.AutoencoderTiny.from_pretrained("./taesd-iono-finetuned")

        # Same schedulers as ConditionedDiffusionModel
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            thresholding=False,
            rescale_betas_zero_snr=False,
            prediction_type=self.hparams.pred_type,
        )
        self.inference_scheduler = diffusers.schedulers.DDPMScheduler(
            thresholding=False,
            rescale_betas_zero_snr=False,
            prediction_type=self.hparams.pred_type,
        )

    def model_loss(self, prediction, x, noise, step):
        """Same loss computation as ConditionedDiffusionModel"""
        if self.hparams.pred_type == 'epsilon':
            target = noise
        elif self.hparams.pred_type == 'v_prediction':
            alpha_t = self.scheduler.alphas_cumprod[step].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            sigma_t = (1 - self.scheduler.alphas_cumprod[step]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            target = alpha_t.sqrt() * noise - sigma_t.sqrt() * x
        else:
            raise ValueError(f"Unknown prediction type: {self.hparams.pred_type}")
        return F.mse_loss(prediction, target)

    def training_step(self, batch, batch_idx):
        """Same training step as ConditionedDiffusionModel"""
        with torch.no_grad():
            images = batch["images"]
            latents_raw = self.vae.encode(scale_to_diffusion(images)).latents
            latents = F.pad(self.unscale_latents(latents_raw), (0, 2, 0, 1))

        # Encode conditioning
        encoded_targets = self.param_encoder(batch["raw_target"])

        # Dropout targets 20% of the time to allow CFG
        use_targets = torch.rand(encoded_targets.size(0), device=encoded_targets.device) < 0.8
        encoded_targets = encoded_targets * use_targets.unsqueeze(1).float()

        # Diffusion forward process
        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)

        # Model prediction
        model_pred = self.model(noisy_latents, steps, class_labels=encoded_targets).sample

        # Compute loss
        loss = self.model_loss(model_pred, latents, noise, steps)
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Same validation step as ConditionedDiffusionModel"""
        with torch.no_grad():
            images = batch["images"]
            latents_raw = self.vae.encode(scale_to_diffusion(images)).latents
            latents = F.pad(self.unscale_latents(latents_raw), (0, 2, 0, 1))

        encoded_targets = self.param_encoder(batch["raw_target"])
        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)
        model_pred = self.model(noisy_latents, steps, class_labels=encoded_targets).sample
        loss = self.model_loss(model_pred, latents, noise, steps)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Same test step as ConditionedDiffusionModel"""
        with torch.no_grad():
            images = batch["images"]
            latents_raw = self.vae.encode(scale_to_diffusion(images)).latents
            latents = F.pad(self.unscale_latents(latents_raw), (0, 2, 0, 1))

        encoded_targets = self.param_encoder(batch["raw_target"])
        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)
        model_pred = self.model(noisy_latents, steps, class_labels=encoded_targets).sample
        loss = self.model_loss(model_pred, latents, noise, steps)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Same optimizer configuration as ConditionedDiffusionModel"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=25,
            eta_min=1e-6,
        )
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        """Same checkpoint generation as ConditionedDiffusionModel"""
        with torch.no_grad():
            x = torch.randn(9, 4, 24, 48, device=self.device)
            encoded_targets = torch.zeros(9, 256, device=self.device)

            self.inference_scheduler.set_timesteps(20, device=self.device)
            for t in self.inference_scheduler.timesteps:
                model_output = self.model(x, t, class_labels=encoded_targets).sample
                step = self.inference_scheduler.step(model_output, t, x)
                x = step.prev_sample

            latents = step.pred_original_sample[..., :23, :46]
            pil_latents = numpy_to_pil(scale_from_diffusion(
                latents[:, :3, ...]).cpu().permute(0, 2, 3, 1).detach().numpy())
            latents = self.scale_latents(latents)
            tv.utils.save_image(
                tv.utils.make_grid(
                    torch.stack([tv.transforms.functional.to_tensor(pil_image) for pil_image in pil_latents]),
                    nrow=3,
                ),
                "out/latents.png",
            ),
            image = self.vae.decode(latents).sample
            image = scale_from_diffusion(image)
            image = image.clip(0.0, 1.0)

            pil_images = numpy_to_pil(image.cpu().permute(0, 2, 3, 1).detach().numpy())

            images = torch.stack([tv.transforms.functional.to_tensor(pil_image.crop((0, 0, 361, 181)))
                                  for pil_image in pil_images])
            image_grid = tv.utils.make_grid(images, nrow=3)

            filename = "out/checkpoint.png"
            tv.utils.save_image(image_grid, filename)
            print(f"Generated images saved to {filename}")

    def scale_latents(self, latents):
        """Scale latents by vae.latent_magnitude for decoding."""
        return latents * self.vae.latent_magnitude

    def unscale_latents(self, latents):
        """Unscale latents by vae.latent_magnitude after encoding."""
        return latents / self.vae.latent_magnitude


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
        images = scale_to_diffusion(batch["images"])  # Scale to [-1, 1]
        latents = self.vae.encode(images).latents
        decoded = self.vae.decode(latents).sample

        img_loss = F.mse_loss(decoded, images)
        loss = img_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images = scale_to_diffusion(batch["images"])  # Scale to [-1, 1]
        latents = self.vae.encode(images).latents
        decoded = self.vae.decode(latents).sample

        img_loss = F.mse_loss(decoded, images)
        loss = img_loss
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = scale_to_diffusion(batch["images"])  # Scale to [-1, 1]
        latents = self.vae.encode(images).latents
        decoded = self.vae.decode(latents).sample

        img_loss = F.mse_loss(decoded, images)
        loss = img_loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def forward(self, images):
        return self.vae.encode(scale_to_diffusion(images)).latents

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
    def pad_map(self, image):
        """Pad the image to 368x184 by wrapping around the columns and repeating the last row."""
        padded_image = image[..., :360].clone()  # Copy the first 360 columns
        # Pad the last 3 rows by repeating the last row
        padded_image = F.pad(padded_image, (0, 0, 0, 3), mode='replicate')
        padded_image = torch.cat((padded_image, padded_image[..., :8]), dim=-1)  # Wrap around the first 8 columns
        return padded_image

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
            self.pad_map,
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
                self.pad_map,
            ])
        else:
            # Don't waste CPU adding a gaussian blur of 0.
            self.augment_train = self.augment_test

        self.century = datetime.date(year=2050, month=1, day=1) - datetime.date(year=1950, month=1, day=1)
        self.midpoint = datetime.datetime(year=2000, month=1, day=1, hour=0, minute=0,
                                          second=0, microsecond=0, tzinfo=datetime.timezone.utc)

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

        secular = torch.tensor([ (dt - self.midpoint) / self.century for dt in dts ], dtype=torch.float32)

        targets = torch.stack([secular, sin_toy, cos_toy, sin_tod, cos_tod,
                               torch.tensor(sample["ssn"]) / 100. - 1.], dim=1)

        raw_targets = torch.stack([secular, toys, tods, torch.tensor(sample["ssn"]) / 100. - 1.], dim=1)
        return {"images": images, "target": targets, "raw_target": raw_targets}

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
