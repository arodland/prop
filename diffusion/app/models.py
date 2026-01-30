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

from abc import ABC, abstractmethod


class DiffusionUtilsMixin:
    """
    Mixin providing common diffusion model utilities.

    This mixin provides shared diffusion loss computation and scheduler
    initialization that can be used by any diffusion model.
    """

    def compute_diffusion_target(self, latents, noise, timesteps, prediction_type='epsilon'):
        """
        Compute the target for diffusion loss based on prediction type.

        This helper method computes what the model should predict given the
        clean latents, noise, and timesteps.

        Args:
            latents: Clean latent representations
            noise: Noise that was added to latents
            timesteps: Diffusion timesteps
            prediction_type: 'epsilon' or 'v_prediction'

        Returns:
            target: What the model should predict
        """
        if prediction_type == 'epsilon':
            return noise
        elif prediction_type == 'v_prediction':
            alpha_t = self.scheduler.alphas_cumprod[timesteps].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            sigma_t = (1 - self.scheduler.alphas_cumprod[timesteps]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            return alpha_t.sqrt() * noise - sigma_t.sqrt() * latents
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

    def compute_standard_diffusion_loss(self, model_pred, latents, noise, timesteps, prediction_type='epsilon'):
        """
        Compute standard MSE diffusion loss.

        This is a shared helper that can be used by any diffusion model.

        Args:
            model_pred: Model's noise/velocity prediction
            latents: Clean latent representations
            noise: Noise that was added to latents
            timesteps: Diffusion timesteps
            prediction_type: 'epsilon' or 'v_prediction'

        Returns:
            loss: Scalar MSE loss tensor
        """
        target = self.compute_diffusion_target(latents, noise, timesteps, prediction_type)
        return F.mse_loss(model_pred, target)

    def create_diffusion_schedulers(self, prediction_type='epsilon'):
        """
        Create training and inference schedulers.

        Helper method to avoid duplicate scheduler initialization.

        Args:
            prediction_type: 'epsilon' or 'v_prediction'

        Returns:
            tuple: (scheduler, inference_scheduler)
        """
        scheduler = diffusers.schedulers.DDPMScheduler(
            thresholding=False,
            rescale_betas_zero_snr=False,
            prediction_type=prediction_type,
        )
        inference_scheduler = diffusers.schedulers.DDPMScheduler(
            thresholding=False,
            rescale_betas_zero_snr=False,
            prediction_type=prediction_type,
        )
        return scheduler, inference_scheduler


class VAEUtilsMixin:
    """
    Mixin providing common VAE encoding/decoding utilities.

    This mixin can be used by any model that works with a VAE to avoid
    code duplication.
    """

    def scale_latents(self, latents):
        """Scale latents by VAE magnitude for decoding."""
        return latents * self.vae.latent_magnitude

    def unscale_latents(self, latents):
        """Unscale latents by VAE magnitude after encoding."""
        return latents / self.vae.latent_magnitude

    def encode_images_to_latents(self, images):
        """
        Encode images to latents using VAE.

        Args:
            images: (B, 3, H, W) images in [0, 1]

        Returns:
            latents: (B, 4, 24, 48) latents ready for diffusion
        """
        with torch.no_grad():
            latents_raw = self.vae.encode(scale_to_diffusion(images)).latents
            latents = F.pad(self.unscale_latents(latents_raw), (0, 2, 0, 1))
        return latents

    def decode_latents_to_images(self, latents):
        """
        Decode latents to images using VAE.

        Args:
            latents: (B, 4, 23, 46) latents from diffusion

        Returns:
            images: (B, 3, H, W) images in [0, 1]
        """
        scaled_latents = self.scale_latents(latents)
        image = self.vae.decode(scaled_latents).sample
        image = scale_from_diffusion(image)
        return image.clip(0.0, 1.0)


class BaseDiffusionModel(DiffusionUtilsMixin, VAEUtilsMixin, L.LightningModule, ABC):
    """
    Base class for diffusion models with common VAE and training infrastructure.

    This class eliminates code duplication by providing:
    - VAE encoding/decoding utilities
    - Shared training/validation/test loop logic
    - Common optimizer and scheduler setup
    - Checkpoint visualization infrastructure

    Subclasses must implement 4 abstract methods:
    - _create_backbone(): Create the denoising model architecture
    - _encode_conditioning(batch): Extract and encode conditioning from batch
    - _compute_loss(model_pred, latents, noise, timesteps): Compute diffusion loss
    - _get_checkpoint_conditioning(): Get conditioning for checkpoint visualization
    """

    def __init__(self,
                 vae_path="./taesd-iono-finetuned",
                 prediction_type='epsilon',
                 learning_rate=1e-4,
                 use_cfg_dropout=False,
                 **kwargs):
        """
        Initialize base diffusion model.

        Args:
            vae_path: Path to pretrained VAE model
            prediction_type: 'epsilon' or 'v_prediction'
            learning_rate: Learning rate for optimizer
            use_cfg_dropout: Whether to apply classifier-free guidance dropout (20%)
        """
        super().__init__()
        self.save_hyperparameters()

        # Load VAE
        self.vae = diffusers.models.AutoencoderTiny.from_pretrained(vae_path)

        # Create diffusion schedulers (using shared helper from DiffusionUtilsMixin)
        self.scheduler, self.inference_scheduler = self.create_diffusion_schedulers(prediction_type)

        # Create backbone (implemented by subclasses)
        self.model = self._create_backbone()

    @abstractmethod
    def _create_backbone(self):
        """
        Create and return the denoising backbone model.

        Subclasses should create their model architecture here (UNet, DiT, etc.)
        and optionally apply torch.compile() and memory format optimizations.

        Returns:
            model: The denoising model
        """
        pass

    @abstractmethod
    def _encode_conditioning(self, batch):
        """
        Encode conditioning information from batch.

        Args:
            batch: Batch dict from dataloader

        Returns:
            tuple: (conditioning_dict, null_conditioning_dict)
                - conditioning_dict: Dict with conditioning tensors (e.g., {'class_labels': tensor})
                - null_conditioning_dict: Dict with null conditioning for CFG dropout

        Example for unconditioned model:
            return {}, {}

        Example for conditioned model:
            encoded = self.param_encoder(batch["raw_target"])
            null = torch.zeros_like(encoded)
            return {"class_labels": encoded}, {"class_labels": null}
        """
        pass

    @abstractmethod
    def _compute_loss(self, model_pred, latents, noise, timesteps):
        """
        Compute diffusion loss given model prediction.

        Args:
            model_pred: Model's noise/velocity prediction
            latents: Clean latent representations
            noise: Noise that was added to latents
            timesteps: Diffusion timesteps

        Returns:
            loss: Scalar loss tensor

        Example for epsilon prediction:
            return F.mse_loss(model_pred, noise)

        Example for v_prediction:
            alpha_t = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sigma_t = 1 - alpha_t
            target = alpha_t.sqrt() * noise - sigma_t.sqrt() * latents
            return F.mse_loss(model_pred, target)

        For standard implementations, use _compute_standard_diffusion_loss() helper.
        """
        pass

    def _compute_standard_diffusion_loss(self, model_pred, latents, noise, timesteps):
        """
        Standard diffusion loss computation for epsilon or v_prediction.

        This is a concrete helper method that delegates to the mixin implementation.
        Subclasses can call this directly from their _compute_loss() implementation.

        Args:
            model_pred: Model's noise/velocity prediction
            latents: Clean latent representations
            noise: Noise that was added to latents
            timesteps: Diffusion timesteps

        Returns:
            loss: Scalar loss tensor
        """
        return self.compute_standard_diffusion_loss(
            model_pred, latents, noise, timesteps, self.hparams.prediction_type
        )

    @abstractmethod
    def _get_checkpoint_conditioning(self):
        """
        Get conditioning for checkpoint visualization (9 samples).

        Returns:
            dict: Conditioning dictionary for 9 samples

        Example for unconditioned model:
            return {}

        Example for parameter sweep:
            params = torch.zeros(9, 4, device=self.device)
            params[:, 2] = torch.linspace(0, 1, 9, device=self.device)
            return {"class_labels": self.param_encoder.rope_enc(params)}
        """
        pass

    def _shared_step(self, batch, is_training=False):
        """
        Shared training/validation/test step logic.

        This is the template method that orchestrates the common diffusion training flow:
        1. Encode images to latents
        2. Get conditioning from batch
        3. Apply CFG dropout if training
        4. Sample noise and timesteps
        5. Add noise to latents
        6. Model forward pass
        7. Compute loss

        Args:
            batch: Batch from dataloader
            is_training: Whether this is a training step (for CFG dropout)

        Returns:
            loss: Scalar loss tensor
        """
        images = batch["images"]

        # Encode to latents
        latents = self.encode_images_to_latents(images)

        # Get conditioning
        conditioning, null_conditioning = self._encode_conditioning(batch)

        # Apply CFG dropout during training (20% of the time, use null conditioning)
        if is_training and self.hparams.use_cfg_dropout:
            use_conditioning = torch.rand(images.size(0), device=self.device) < 0.8
            # Apply dropout by replacing with null conditioning
            for key in conditioning:
                if key in null_conditioning:
                    # Broadcast use_conditioning to match conditioning shape
                    mask = use_conditioning.view(-1, *([1] * (conditioning[key].ndim - 1)))
                    conditioning[key] = torch.where(
                        mask,
                        conditioning[key],
                        null_conditioning[key]
                    )

        # Diffusion forward process
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            self.scheduler.config.num_train_timesteps,
            (images.size(0),),
            device=self.device
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Model prediction
        model_pred = self.model(noisy_latents, timesteps, **conditioning).sample

        # Compute loss
        loss = self._compute_loss(model_pred, latents, noise, timesteps)

        return loss

    def training_step(self, batch, batch_idx):
        """Training step - uses shared logic with CFG dropout."""
        loss = self._shared_step(batch, is_training=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - uses shared logic without CFG dropout."""
        loss = self._shared_step(batch, is_training=False)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step - uses shared logic without CFG dropout."""
        loss = self._shared_step(batch, is_training=False)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        learning_rate = getattr(self.hparams, 'learning_rate', 1e-4)
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=1.0)
        return [optimizer], [scheduler]

    def _encode_checkpoint_params(self, raw_params):
        """
        Encode raw parameters for checkpoint generation.

        This method can be overridden by subclasses to customize parameter encoding
        for checkpoint visualization.

        Args:
            raw_params: (9, 4) raw parameter tensor

        Returns:
            dict: Conditioning dictionary (e.g., {"class_labels": encoded})
        """
        # Default: delegate to _get_checkpoint_conditioning()
        return self._get_checkpoint_conditioning()

    def _generate_checkpoint_from_validation(self, conditioning_key="class_labels"):
        """
        Generate checkpoint images using random validation examples.

        This is a helper for models that want to sample conditioning from the
        validation dataset rather than using synthetic conditioning.

        Args:
            conditioning_key: Key to use in conditioning dict (default: "class_labels")

        Returns:
            None (saves images to disk)
        """
        with torch.no_grad():
            # Sample random validation examples for conditioning
            val_dataloader = self.trainer.datamodule.val_dataloader()
            val_dataset = val_dataloader.dataset

            # Sample 9 random indices
            indices = torch.randperm(len(val_dataset))[:9]
            raw_targets = []
            for idx in indices:
                sample = val_dataset[idx.item()]
                raw_targets.append(sample["raw_target"])

            # Stack and move to device
            raw_targets_batch = torch.stack(raw_targets).to(self.device)

            # Encode using subclass-specific method
            conditioning = self._encode_checkpoint_params(raw_targets_batch)

            # Generate images
            x = torch.randn(9, 4, 24, 48, device=self.device)
            self.inference_scheduler.set_timesteps(20, device=self.device)
            for t in self.inference_scheduler.timesteps:
                model_output = self.model(x, t, **conditioning).sample
                step = self.inference_scheduler.step(model_output, t, x)
                x = step.prev_sample

            latents = step.pred_original_sample[..., :23, :46]
            pil_latents = numpy_to_pil(scale_from_diffusion(
                latents[:, :3, ...]).cpu().permute(0, 2, 3, 1).detach().numpy())

            # Save latent visualization
            tv.utils.save_image(
                tv.utils.make_grid(
                    torch.stack([tv.transforms.functional.to_tensor(pil_image) for pil_image in pil_latents]),
                    nrow=3,
                ),
                "out/latents.png",
            )

            # Decode to images
            image = self.decode_latents_to_images(latents)
            pil_images = numpy_to_pil(image.cpu().permute(0, 2, 3, 1).detach().numpy())

            images = torch.stack([tv.transforms.functional.to_tensor(pil_image.crop((0, 0, 361, 181)))
                                  for pil_image in pil_images])
            image_grid = tv.utils.make_grid(images, nrow=3)

            filename = "out/checkpoint.png"
            tv.utils.save_image(image_grid, filename)
            print(f"Generated images saved to {filename}")

    def on_save_checkpoint(self, checkpoint):
        """Generate sample images for checkpoint visualization."""
        with torch.no_grad():
            # Get conditioning for 9 samples
            conditioning = self._get_checkpoint_conditioning()

            # Generate latents through denoising process
            x = torch.randn(9, 4, 24, 48, device=self.device)
            self.inference_scheduler.set_timesteps(20, device=self.device)
            for t in self.inference_scheduler.timesteps:
                model_output = self.model(x, t, **conditioning).sample
                step = self.inference_scheduler.step(model_output, t, x)
                x = step.prev_sample

            # Decode to images
            latents = step.pred_original_sample[..., :23, :46]

            # Save latent visualization (first 3 channels)
            pil_latents = numpy_to_pil(scale_from_diffusion(
                latents[:, :3, ...]).cpu().permute(0, 2, 3, 1).detach().numpy())
            tv.utils.save_image(
                tv.utils.make_grid(
                    torch.stack([tv.transforms.functional.to_tensor(pil_image)
                                for pil_image in pil_latents]),
                    nrow=3,
                ),
                "out/latents.png",
            )

            # Decode to full images
            image = self.decode_latents_to_images(latents)
            pil_images = numpy_to_pil(image.cpu().permute(0, 2, 3, 1).detach().numpy())

            # Crop to original size and create grid
            images = torch.stack([tv.transforms.functional.to_tensor(pil_image.crop((0, 0, 361, 181)))
                                 for pil_image in pil_images])
            image_grid = tv.utils.make_grid(images, nrow=3)

            filename = "out/checkpoint.png"
            tv.utils.save_image(image_grid, filename)
            print(f"Generated images saved to {filename}")


class DiffusionModel(BaseDiffusionModel):
    """Unconditional diffusion model using UNet backbone."""

    def __init__(self):
        super().__init__(
            vae_path="./taesd-iono-finetuned",
            prediction_type='epsilon',
            learning_rate=5e-5,
            use_cfg_dropout=False,
        )

    def _create_backbone(self):
        """Create UNet2D backbone with attention blocks."""
        model = diffusers.models.UNet2DModel(
            sample_size=(24, 48),
            in_channels=4,
            out_channels=4,
            block_out_channels=(64, 128, 256, 512),
            dropout=0.1,
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
        )
        model.to(memory_format=torch.channels_last)
        model = torch.compile(model)
        return model

    def _encode_conditioning(self, batch):
        """No conditioning for unconditional diffusion."""
        return {}, {}

    def _compute_loss(self, model_pred, latents, noise, timesteps):
        """Simple epsilon prediction loss."""
        return F.mse_loss(model_pred, noise)

    def _get_checkpoint_conditioning(self):
        """No conditioning for checkpoint generation."""
        return {}

    def configure_optimizers(self):
        """Override to use CosineAnnealingWarmRestarts scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=25,
            eta_min=1e-6,
        )
        return [optimizer], [scheduler]

class RopeEncoder(L.LightningModule):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        freq = (500 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freq", freq)

        self.scrambler = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),  # Preserves negatives, unlike ReLU
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
    Embeds timesteps using sinusoidal encoding.
    Returns frequency_embedding_size-dimensional embeddings (default 256).
    """
    def __init__(self, hidden_size=768, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element, or a scalar.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # Handle scalar timesteps (e.g., during checkpoint generation)
        if t.ndim == 0:
            t = t.unsqueeze(0)

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
        return self.timestep_embedding(t, self.frequency_embedding_size)


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


class MultiheadSDPA(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention using F.scaled_dot_product_attention.
    Automatically uses flash attention when available (PyTorch 2.0+, CUDA capability >= 8.0).
    """
    def __init__(self, embed_dim, num_heads, batch_first=True, bias=True):
        super().__init__()
        assert batch_first, "Only batch_first=True is supported"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # QKV projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        """
        Args:
            query: (B, N_q, embed_dim)
            key: (B, N_kv, embed_dim)
            value: (B, N_kv, embed_dim)
            key_padding_mask: (B, N_kv) boolean mask where True means ignore
            need_weights: ignored (for compatibility)
            attn_mask: ignored (for compatibility)

        Returns:
            Tuple of (output, None) to match nn.MultiheadAttention API
        """
        B, N_q, _ = query.shape
        N_kv = key.shape[1]

        # Project and reshape to (B, num_heads, N, head_dim)
        q = self.q_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Convert key_padding_mask from (B, N_kv) to (B, 1, 1, N_kv) for broadcasting
        attn_mask_sdpa = None
        if key_padding_mask is not None:
            # True in key_padding_mask means "ignore this position"
            # SDPA expects attn_mask where True means "keep", so we invert
            attn_mask_sdpa = ~key_padding_mask.view(B, 1, 1, N_kv)

        # Flash attention via scaled_dot_product_attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask_sdpa,
            dropout_p=0.0,
            is_causal=False,
        )

        # Reshape back: (B, num_heads, N_q, head_dim) -> (B, N_q, embed_dim)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.embed_dim)
        out = self.out_proj(out)

        return out, None  # Return None for weights to match API


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


class ConditionedDiffusionModel(BaseDiffusionModel):
    """Diffusion model conditioned on global parameters (secular, toy, tod, ssn)."""

    def __init__(self, pred_type='epsilon'):
        super().__init__(
            vae_path="./taesd-iono-finetuned",
            prediction_type=pred_type,
            learning_rate=1e-4,
            use_cfg_dropout=True,  # Enable 20% CFG dropout
        )

        # Parameter encoder for conditioning
        self.param_encoder = RopeEncoder(dim=64)

    def _create_backbone(self):
        """Create UNet2D backbone with class conditioning."""
        model = diffusers.models.UNet2DModel(
            sample_size=(24, 48),
            in_channels=4,
            out_channels=4,
            block_out_channels=(64, 128, 256, 512),
            dropout=0.1,
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            class_embed_type="identity",
        )
        model.to(memory_format=torch.channels_last)
        model = torch.compile(model)
        return model

    def _encode_conditioning(self, batch):
        """Encode parameters for conditioning with CFG dropout support."""
        params = batch.get("raw_target")
        if params is None:
            # Fallback for batches without params
            params = torch.zeros(batch["images"].shape[0], 4, device=self.device)

        encoded = self.param_encoder(params)
        null_encoded = torch.zeros_like(encoded)

        return {"class_labels": encoded}, {"class_labels": null_encoded}

    def _compute_loss(self, model_pred, latents, noise, timesteps):
        """Use standard diffusion loss computation."""
        return self._compute_standard_diffusion_loss(model_pred, latents, noise, timesteps)

    def _get_checkpoint_conditioning(self):
        """Not used - overridden by _encode_checkpoint_params."""
        return {"class_labels": torch.zeros(9, 256, device=self.device)}

    def _encode_checkpoint_params(self, raw_params):
        """Encode parameters using full param_encoder (with scrambler)."""
        encoded_targets = self.param_encoder(raw_params)
        return {"class_labels": encoded_targets}

    def on_save_checkpoint(self, checkpoint):
        """Generate images using random validation set conditioning."""
        self._generate_checkpoint_from_validation()


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

        # Timestep embedder (sinusoidal encoding only, no projection)
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=class_embed_size)

        # Conditioning projector: concatenate [timestep, class] then project
        # Input: 256 (timestep) + 256 (class) = 512
        # Output: 768 (hidden_size)
        self.conditioning_projector = nn.Sequential(
            nn.Linear(class_embed_size * 2, hidden_size),
            nn.SiLU(),
        )

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
            timesteps: (batch,) timestep indices, or scalar
            class_labels: (batch, 256) continuous class embeddings (optional)
        Returns:
            (batch, 4, 24, 48) predicted noise/velocity
        """
        batch_size = x.shape[0]

        # Handle scalar or mismatched batch timesteps
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.shape[0] != batch_size:
            timesteps = timesteps.expand(batch_size)

        # Patchify: (B, 4, 24, 48) → (B, 768, 12, 24)
        x = self.x_embedder(x)
        # Flatten: (B, 768, 12, 24) → (B, 288, 768)
        x = x.flatten(2).transpose(1, 2)

        # Add positional embeddings
        x = x + self.pos_embed

        # Get timestep embeddings (256-dim sinusoidal)
        t = self.t_embedder(timesteps)  # (B, 256)

        # Get class embeddings (256-dim from RopeEncoder)
        if class_labels is not None:
            y = class_labels  # (B, 256)
        else:
            y = torch.zeros_like(t)

        # Combine conditioning: concatenate then project
        # [timestep, class] -> (B, 512) -> (B, 768)
        c = torch.cat([t, y], dim=1)  # (B, 512)
        c = self.conditioning_projector(c)  # (B, 768)

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


class DiTDiffusionModel(BaseDiffusionModel):
    """
    DiT-based diffusion model (drop-in replacement for ConditionedDiffusionModel).

    Uses Diffusion Transformer (DiT-B/2) instead of UNet backbone.
    """

    def __init__(self, pred_type='epsilon'):
        super().__init__(
            vae_path="./taesd-iono-finetuned",
            prediction_type=pred_type,
            learning_rate=1e-4,
            use_cfg_dropout=True,  # Enable 20% CFG dropout
        )

        # Same param encoder as ConditionedDiffusionModel
        self.param_encoder = RopeEncoder(dim=64)

    def _create_backbone(self):
        """Create DiT-B/2 backbone with continuous class conditioning."""
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
        model = ModelOutputWrapper(dit_backbone)
        model = torch.compile(model)
        return model

    def _encode_conditioning(self, batch):
        """Encode parameters using RoPE (without scrambler) for conditioning."""
        params = batch.get("raw_target")
        if params is None:
            params = torch.zeros(batch["images"].shape[0], 4, device=self.device)

        # Use raw RoPE embeddings (skip scrambler for DiT)
        encoded = self.param_encoder.rope_enc(params)
        null_encoded = torch.zeros_like(encoded)

        return {"class_labels": encoded}, {"class_labels": null_encoded}

    def _compute_loss(self, model_pred, latents, noise, timesteps):
        """Use standard diffusion loss computation."""
        return self._compute_standard_diffusion_loss(model_pred, latents, noise, timesteps)

    def _get_checkpoint_conditioning(self):
        """Not used - overridden by _encode_checkpoint_params."""
        return {"class_labels": torch.zeros(9, 256, device=self.device)}

    def _encode_checkpoint_params(self, raw_params):
        """Encode parameters using RoPE encoder (without scrambler)."""
        encoded_targets = self.param_encoder.rope_enc(raw_params)
        return {"class_labels": encoded_targets}

    def on_save_checkpoint(self, checkpoint):
        """Generate images using random validation set conditioning."""
        self._generate_checkpoint_from_validation()


class ObservationEncoder(nn.Module):
    """
    Encodes sparse observations for cross-attention conditioning.

    Each observation consists of:
    - Location: (lat, lon) normalized to [-1, 1]
    - Values: (fof2, mufd, hmf2) normalized to [0, 1]
    - Weights: (weight_fof2, weight_mufd, weight_hmf2) - 0 if absent, confidence score [0,1] if present
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim

        # Encode location (lat, lon normalized to [-1, 1])
        self.location_embed = nn.Linear(2, embed_dim // 2)

        # Encode each channel with its weight
        # Each channel gets: (value, weight) where weight is 0 or confidence score
        self.channel_encoders = nn.ModuleList([
            nn.Linear(2, embed_dim // 6) for _ in range(3)  # fof2, mufd, hmf2
        ])

        # Final projection
        self.proj = nn.Linear(embed_dim // 2 + 3 * (embed_dim // 6), embed_dim)

    def forward(self, observations):
        """
        Args:
            observations: (B, N, 8) tensor of:
                [lat, lon, fof2, mufd, hmf2, weight_fof2, weight_mufd, weight_hmf2]
                where weight_* is 0 if channel absent, or confidence score [0, 1] if present

        Returns:
            (B, N, embed_dim) observation embeddings
        """
        B, N, _ = observations.shape

        # Extract components
        locations = observations[..., :2]  # (B, N, 2)
        values = observations[..., 2:5]    # (B, N, 3)
        weights = observations[..., 5:8]   # (B, N, 3)

        # Encode location
        loc_embed = self.location_embed(locations)  # (B, N, embed_dim//2)

        # Encode each channel with its weight
        channel_embeds = []
        for i, encoder in enumerate(self.channel_encoders):
            # Concatenate value and weight
            ch_input = torch.stack([
                values[..., i],
                weights[..., i]
            ], dim=-1)  # (B, N, 2)
            ch_embed = encoder(ch_input)  # (B, N, embed_dim//6)
            channel_embeds.append(ch_embed)

        # Concatenate all embeddings
        all_embeds = torch.cat([loc_embed] + channel_embeds, dim=-1)

        # Final projection
        return self.proj(all_embeds)  # (B, N, embed_dim)


class DiTBlockWithObservations(nn.Module):
    """
    DiT block with cross-attention to observations.
    Adds observation cross-attention after self-attention.
    Uses MultiheadSDPA for flash attention support on cross-attention.
    """
    def __init__(self, hidden_size=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

        # Cross-attention to observations (uses flash attention)
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiheadSDPA(hidden_size, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )

        # AdaLN modulation (same as DiTBlock)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Initialize modulation to zero
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c, obs_embeds=None, obs_weights=None):
        """
        Args:
            x: (B, N_patches, hidden_size) - spatial tokens
            c: (B, hidden_size) - conditioning (timestep + params)
            obs_embeds: (B, N_obs, hidden_size) - observation embeddings (optional)
            obs_weights: (B, N_obs, 3) - per-channel weights [0, 1] for soft masking (optional)

        Returns:
            (B, N_patches, hidden_size) - updated tokens
        """
        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-attention
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(x_norm, x_norm, x_norm)[0]

        # Cross-attention to observations (if provided and not all zero)
        if obs_embeds is not None:
            # Soft-weight the observation embeddings
            if obs_weights is not None:
                # Use max weight across channels as overall observation weight
                overall_weight = obs_weights.max(dim=-1)[0]  # (B, N_obs)

                # Skip cross-attention if all observations have zero weight
                # This handles the case for observation guidance with zero weights
                if overall_weight.sum() > 0:
                    # Scale embeddings by weights for soft masking
                    weighted_obs_embeds = obs_embeds * overall_weight.unsqueeze(-1)
                    # Only mask out completely zero-weight observations (all channels zero)
                    key_padding_mask = (overall_weight == 0.0)

                    x_cross = self.norm_cross(x)
                    cross_out = self.cross_attn(
                        x_cross,
                        weighted_obs_embeds,
                        weighted_obs_embeds,
                        key_padding_mask=key_padding_mask
                    )[0]
                    x = x + cross_out
            else:
                # No weights provided, use observations as-is
                weighted_obs_embeds = obs_embeds
                key_padding_mask = None

                x_cross = self.norm_cross(x)
                cross_out = self.cross_attn(
                    x_cross,
                    weighted_obs_embeds,
                    weighted_obs_embeds,
                    key_padding_mask=key_padding_mask
                )[0]
                x = x + cross_out

        # MLP
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class ObservationConditionedDiT(DiffusionUtilsMixin, VAEUtilsMixin, L.LightningModule):
    """
    DiT model conditioned on both global parameters and sparse observations.

    Uses cross-attention to observation embeddings to learn spatial interpolation.
    Handles missing channels per observation and variable observation counts.
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
        max_observations=100,
        pred_type="v_prediction",
    ):
        super().__init__()
        self.save_hyperparameters()

        # VAE (same as DiT)
        self.vae = diffusers.models.AutoencoderTiny.from_pretrained("./taesd-iono-finetuned")
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        self.vae.latent_magnitude = 4.0

        # Parameter encoder (same as DiT - uses RoPE without scrambler)
        self.param_encoder = RopeEncoder(dim=64)

        # Observation encoder
        self.obs_encoder = ObservationEncoder(embed_dim=hidden_size)

        # Patchify
        self.input_h, self.input_w = input_size
        self.num_patches_h = self.input_h // patch_size
        self.num_patches_w = self.input_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.x_embedder = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size, bias=True
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=class_embed_size)

        # Conditioning projector: concatenate [timestep, class] then project
        self.conditioning_projector = nn.Sequential(
            nn.Linear(class_embed_size * 2, hidden_size),
            nn.SiLU(),
        )

        # DiT blocks with cross-attention to observations
        self.blocks = nn.ModuleList([
            DiTBlockWithObservations(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

        # Scheduler (using shared helper from DiffusionUtilsMixin)
        self.scheduler, self.inference_scheduler = self.create_diffusion_schedulers(pred_type)

        # Initialize weights
        self.initialize_weights()

        # Compile model for speed
        self.model_forward = torch.compile(self.forward_model)

    def initialize_weights(self):
        # Initialize patch embedding
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_model(self, x, timesteps, class_labels, obs_embeds=None, obs_weights=None):
        """
        Forward pass through the DiT model.

        Args:
            x: (B, 4, 24, 48) noisy latents
            timesteps: (B,) timestep indices
            class_labels: (B, 256) continuous class embeddings
            obs_embeds: (B, N, 768) observation embeddings (optional)
            obs_weights: (B, N, 3) per-channel observation weights [0, 1] (optional)

        Returns:
            (B, 4, 24, 48) predicted noise/velocity
        """
        batch_size = x.shape[0]

        # Handle scalar timesteps
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.shape[0] != batch_size:
            timesteps = timesteps.expand(batch_size)

        # Patchify and add position embeddings
        x = self.x_embedder(x)  # (B, hidden_size, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        x = x + self.pos_embed  # (B, num_patches, hidden_size)

        # Compute conditioning
        t_emb = self.t_embedder(timesteps)  # (B, 256)

        # Ensure class_labels is not None
        if class_labels is None:
            class_labels = torch.zeros(batch_size, 256, device=x.device)

        # Concatenate and project conditioning
        c = torch.cat([t_emb, class_labels], dim=1)  # (B, 512)
        c = self.conditioning_projector(c)  # (B, hidden_size)

        # Apply transformer blocks with observation cross-attention
        for block in self.blocks:
            x = block(x, c, obs_embeds, obs_weights)

        # Final layer (includes unpatchify)
        x = self.final_layer(x, c, self.num_patches_h, self.num_patches_w)  # (B, in_channels, H, W)

        return x

    def model_loss(self, prediction, x, noise, step):
        """Compute diffusion loss using shared helper from DiffusionUtilsMixin"""
        return self.compute_standard_diffusion_loss(prediction, x, noise, step, self.hparams.pred_type)

    def sample_observations_varied(self, images):
        """
        Vectorized observation sampling for speed.
        Runs entirely on GPU to avoid CPU bottleneck.

        Uses cosine-weighted latitude sampling to avoid over-sampling poles.
        Per-channel weights: 0 if absent, confidence score if present.
        """
        B, C, H, W = images.shape
        device = images.device
        max_obs = self.hparams.max_observations

        observations = torch.zeros(B, max_obs, 8, device=device)
        obs_locations = torch.zeros(B, max_obs, 2, dtype=torch.long, device=device)

        # Vectorized: sample number of observations for each batch element
        scenarios = torch.rand(B, device=device)
        num_obs_list = torch.zeros(B, dtype=torch.long, device=device)

        # 20%: Very sparse (10-20 obs)
        num_obs_list[scenarios < 0.20] = torch.randint(10, 21, (1,), device=device).item()
        # 40%: Realistic (20-35 obs)
        num_obs_list[(scenarios >= 0.20) & (scenarios < 0.60)] = torch.randint(20, 36, (1,), device=device).item()
        # 40%: Dense (40-60 obs)
        num_obs_list[scenarios >= 0.60] = torch.randint(40, min(61, max_obs + 1), (1,), device=device).item()

        # Sample locations with cosine-weighted latitude distribution
        # Latitude sampling: proportional to cos(latitude) to match sphere area
        # Pixel y=0 is -90°, y=90 is 0°, y=180 is +90°
        # Sample u ~ Uniform(0, 1), then lat_degrees = arcsin(2u - 1)
        u = torch.rand(B, max_obs, device=device)
        # Clamp to avoid arcsin(x) with |x| > 1 due to floating point errors
        lat_input = (2 * u - 1).clamp(-1.0, 1.0)
        lat_degrees = torch.asin(lat_input) * (180 / math.pi)  # Result in [-90, 90]
        # Convert to pixel coordinates: y = (lat_degrees + 90) / 180 * H
        all_lats = ((lat_degrees + 90) / 180 * H).long().clamp(0, H - 1)

        # Longitude sampling: uniform (already correct for sphere)
        all_lons = torch.randint(0, W, (B, max_obs), device=device)

        # Create mask for valid observations
        obs_indices = torch.arange(max_obs, device=device).unsqueeze(0).expand(B, -1)
        valid_mask = obs_indices < num_obs_list.unsqueeze(1)

        # Store locations
        obs_locations[:, :, 0] = all_lats
        obs_locations[:, :, 1] = all_lons

        # Normalize locations to [-1, 1]
        observations[:, :, 0] = (all_lats.float() / H) * 2 - 1
        observations[:, :, 1] = (all_lons.float() / W) * 2 - 1

        # Sample confidence weights (biased toward higher values)
        # Use Beta(2, 1) distribution: more density near 1.0
        confidence_scores = torch.distributions.Beta(2.0, 1.0).sample((B, max_obs)).to(device)

        # Vectorized: determine channel availability
        # 60% all channels, 25% two channels, 15% one channel
        channel_scenarios = torch.rand(B, max_obs, device=device)

        # All channels (60%)
        all_channels_mask = channel_scenarios < 0.60
        # Two channels (25%)
        two_channels_mask = (channel_scenarios >= 0.60) & (channel_scenarios < 0.85)
        # One channel (15%)
        one_channel_mask = channel_scenarios >= 0.85

        # Gather observation values from images (vectorized)
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, max_obs)
        for ch in range(3):
            values = images[batch_indices, ch, all_lats, all_lons]
            observations[:, :, 2 + ch] = values

            # Set weights: 0 if absent, confidence_score if present
            # All channels: all 3 present
            observations[:, :, 5 + ch][all_channels_mask] = confidence_scores[all_channels_mask]

            # Two channels: randomly drop one
            if two_channels_mask.any():
                drop_ch = torch.randint(0, 3, (1,), device=device).item()
                if ch == drop_ch:
                    observations[:, :, 5 + ch][two_channels_mask] = 0.0
                else:
                    observations[:, :, 5 + ch][two_channels_mask] = confidence_scores[two_channels_mask]

            # One channel: randomly keep one
            if one_channel_mask.any():
                keep_ch = torch.randint(0, 3, (1,), device=device).item()
                if ch == keep_ch:
                    observations[:, :, 5 + ch][one_channel_mask] = confidence_scores[one_channel_mask]
                else:
                    observations[:, :, 5 + ch][one_channel_mask] = 0.0

        # Zero out values for absent channels
        for ch in range(3):
            observations[:, :, 2 + ch] *= (observations[:, :, 5 + ch] > 0).float()

        # Zero out invalid observations (beyond num_obs for each batch)
        observations[~valid_mask] = 0.0

        return observations, obs_locations


    def compute_observation_loss(self, model_pred, noisy_latents, steps, observations, obs_locations, target_latents):
        """
        Vectorized computation of weighted MSE loss at observation locations in latent space.
        Weights observations by their confidence scores.

        Args:
            model_pred: (B, 4, 24, 48) model's prediction
            noisy_latents: (B, 4, 24, 48) noisy latents
            steps: (B,) timesteps
            observations: (B, N, 8) observation data with per-channel weights
            obs_locations: (B, N, 2) pixel locations
            target_latents: (B, 4, 24, 48) ground truth latents
        """
        B, N = observations.shape[:2]

        # Extract predicted clean latent
        if self.hparams.pred_type == 'epsilon':
            alpha_t = self.scheduler.alphas_cumprod[steps].view(-1, 1, 1, 1)
            predicted_latent = (noisy_latents - (1 - alpha_t).sqrt() * model_pred) / alpha_t.sqrt()
        elif self.hparams.pred_type == 'v_prediction':
            alpha_t = self.scheduler.alphas_cumprod[steps].view(-1, 1, 1, 1)
            sigma_t = 1 - alpha_t
            predicted_latent = alpha_t.sqrt() * noisy_latents - sigma_t.sqrt() * model_pred
        else:
            raise ValueError(f"Unknown prediction type: {self.hparams.pred_type}")

        # Extract per-channel observation weights and compute mean weight per observation
        channel_weights = observations[..., 5:8]  # (B, N, 3)
        obs_weights = channel_weights.mean(dim=-1)  # (B, N) - mean of channel weights

        # Check which observations are valid (have at least one channel with non-zero weight)
        obs_mask = channel_weights.sum(dim=-1) > 0  # (B, N)

        # Convert pixel locations to latent locations (vectorized)
        latent_lats = (obs_locations[..., 0].float() / 181.0 * 23.0).long().clamp(0, 22)
        latent_lons = (obs_locations[..., 1].float() / 361.0 * 46.0).long().clamp(0, 45)

        # Gather predicted and target latents at observation locations (vectorized)
        batch_indices = torch.arange(B, device=predicted_latent.device).unsqueeze(1).expand(-1, N)

        # predicted_latent shape: (B, 4, 24, 48)
        # We want to gather at (batch_indices, :, latent_lats, latent_lons)
        # Expand for all latent channels
        batch_idx_exp = batch_indices.unsqueeze(1).expand(-1, 4, -1)  # (B, 4, N)
        channel_idx = torch.arange(4, device=predicted_latent.device).view(1, 4, 1).expand(B, -1, N)
        lat_idx = latent_lats.unsqueeze(1).expand(-1, 4, -1)  # (B, 4, N)
        lon_idx = latent_lons.unsqueeze(1).expand(-1, 4, -1)  # (B, 4, N)

        # Gather using advanced indexing
        pred_at_obs = predicted_latent[batch_idx_exp, channel_idx, lat_idx, lon_idx]  # (B, 4, N)
        target_at_obs = target_latents[batch_idx_exp, channel_idx, lat_idx, lon_idx]  # (B, 4, N)

        # Compute squared error
        squared_error = (pred_at_obs - target_at_obs) ** 2  # (B, 4, N)

        # Average over latent channels
        squared_error = squared_error.mean(dim=1)  # (B, N)

        # Weight errors by observation confidence
        weighted_errors = squared_error * obs_weights  # (B, N)

        # Apply observation mask and compute weighted mean
        valid_errors = weighted_errors[obs_mask]
        valid_weights = obs_weights[obs_mask]

        if valid_errors.numel() == 0:
            # Return zero loss with gradient support
            return (predicted_latent * 0.0).mean()

        # Normalize by sum of weights
        return valid_errors.sum() / (valid_weights.sum() + 1e-8)

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        params = batch["raw_target"]  # [secular, toy, tod, ssn]

        # SSN corruption (30% of batches)
        corrupted_params = params.clone()
        corruption_type = torch.rand(1).item()

        if corruption_type < 0.15:  # 15%: Zero out SSN
            corrupted_params[:, 3] = 0.0
        elif corruption_type < 0.30:  # 15%: Add noise to SSN
            ssn_noise = torch.randn_like(corrupted_params[:, 3]) * 0.3
            corrupted_params[:, 3] = corrupted_params[:, 3] + ssn_noise

        # Sample observations
        observations, obs_locations = self.sample_observations_varied(images)

        # Add measurement noise (increases over training)
        noise_schedule = min(0.05, 0.005 + self.global_step / 200000 * 0.045)
        obs_noise = torch.randn_like(observations[..., 2:5]) * noise_schedule
        observations[..., 2:5] = observations[..., 2:5] + obs_noise * observations[..., 5:8]

        # Encode images to latents
        latents = self.encode_images_to_latents(images)

        # Encode parameters and observations
        param_embeds = self.param_encoder.rope_enc(corrupted_params)
        obs_embeds = self.obs_encoder(observations)
        obs_weights = observations[..., 5:8]  # Per-channel weights [0, 1]

        # Diffusion forward process
        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)

        # Model prediction with observations
        model_pred = self.model_forward(noisy_latents, steps, param_embeds, obs_embeds, obs_weights)

        # Combined loss
        diffusion_loss = self.model_loss(model_pred, latents, noise, steps)

        # Observation fitting loss (weighted by denoising progress)
        progress = 1.0 - steps.float() / self.scheduler.config.num_train_timesteps
        obs_weight = (progress ** 2).mean()

        obs_loss = self.compute_observation_loss(
            model_pred, noisy_latents, steps, observations, obs_locations, latents
        )

        total_loss = diffusion_loss + 10.0 * obs_weight * obs_loss

        # Get current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']

        self.log_dict({
            "train_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "obs_loss": obs_loss,
            "obs_weight": obs_weight,
            "obs_noise_std": noise_schedule,
            "lr": current_lr,
        }, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation with perfect observations (no corruption/noise)"""
        images = batch["images"]
        params = batch["raw_target"]

        # Sample observations (no corruption)
        observations, obs_locations = self.sample_observations_varied(images)

        # Encode
        latents = self.encode_images_to_latents(images)

        param_embeds = self.param_encoder.rope_enc(params)
        obs_embeds = self.obs_encoder(observations)
        obs_weights = observations[..., 5:8]  # Per-channel weights [0, 1]

        # Diffusion
        noise = torch.randn_like(latents)
        steps = torch.randint(self.scheduler.config.num_train_timesteps, (images.size(0),), device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, steps)

        model_pred = self.model_forward(noisy_latents, steps, param_embeds, obs_embeds, obs_weights)

        loss = self.model_loss(model_pred, latents, noise, steps)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        # Exponential decay from 1e-4 to 1e-5 over 200 epochs
        # gamma = 10^(-1/200) ≈ 0.98855 so that lr_200 = 1e-4 * gamma^200 = 1e-5
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98855)
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        """Generate images using random validation set conditioning with observations"""
        with torch.no_grad():
            # Sample random validation examples
            val_dataloader = self.trainer.datamodule.val_dataloader()
            val_dataset = val_dataloader.dataset

            # Sample 9 random indices
            indices = torch.randperm(len(val_dataset))[:9]
            raw_targets = []
            images_list = []
            for idx in indices:
                sample = val_dataset[idx.item()]
                raw_targets.append(sample["raw_target"])
                images_list.append(sample["images"])

            # Stack and move to device
            raw_targets_batch = torch.stack(raw_targets).to(self.device)
            images_batch = torch.stack(images_list).to(self.device)

            # Sample observations from these images (realistic count: 25 observations)
            observations, obs_locations = self.sample_observations_from_images(
                images_batch, num_obs=25
            )

            # Encode parameters and observations
            param_embeds = self.param_encoder.rope_enc(raw_targets_batch)
            obs_embeds = self.obs_encoder(observations)
            obs_weights = observations[..., 5:8]  # Per-channel weights [0, 1]

            # Generate images
            x = torch.randn(9, 4, 24, 48, device=self.device)
            self.inference_scheduler.set_timesteps(50, device=self.device)
            for t in self.inference_scheduler.timesteps:
                model_output = self.model_forward(x, t, param_embeds, obs_embeds, obs_weights)
                step = self.inference_scheduler.step(model_output, t, x)
                x = step.prev_sample

            # Decode latents to images
            latents = step.pred_original_sample[..., :23, :46]
            pil_latents = numpy_to_pil(scale_from_diffusion(
                latents[:, :3, ...]).cpu().permute(0, 2, 3, 1).detach().numpy())
            latents = self.scale_latents(latents)
            tv.utils.save_image(
                tv.utils.make_grid(
                    torch.stack([tv.transforms.functional.to_tensor(pil_image) for pil_image in pil_latents]),
                    nrow=3,
                ),
                "out/obs_latents.png",
            )

            image = self.vae.decode(latents).sample
            image = scale_from_diffusion(image)
            image = image.clip(0.0, 1.0)

            pil_images = numpy_to_pil(image.cpu().permute(0, 2, 3, 1).detach().numpy())

            images = torch.stack([tv.transforms.functional.to_tensor(pil_image.crop((0, 0, 361, 181)))
                                  for pil_image in pil_images])
            image_grid = tv.utils.make_grid(images, nrow=3)

            filename = "out/obs_checkpoint.png"
            tv.utils.save_image(image_grid, filename)
            print(f"Generated observation-conditioned images saved to {filename}")

    def on_load_checkpoint(self, checkpoint):
        """Migrate old nn.MultiheadAttention weights to new MultiheadSDPA format"""
        state_dict = checkpoint['state_dict']
        needs_migration = any('cross_attn.in_proj_weight' in k for k in state_dict.keys())

        if needs_migration:
            print("Migrating checkpoint from nn.MultiheadAttention to MultiheadSDPA format...")
            for i in range(len(self.blocks)):
                prefix = f'blocks.{i}.cross_attn'
                in_proj_weight_key = f'{prefix}.in_proj_weight'
                in_proj_bias_key = f'{prefix}.in_proj_bias'

                if in_proj_weight_key in state_dict:
                    # Split combined in_proj into separate q, k, v projections
                    in_proj_weight = state_dict.pop(in_proj_weight_key)
                    embed_dim = in_proj_weight.shape[1]
                    q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)

                    state_dict[f'{prefix}.q_proj.weight'] = q_weight
                    state_dict[f'{prefix}.k_proj.weight'] = k_weight
                    state_dict[f'{prefix}.v_proj.weight'] = v_weight

                if in_proj_bias_key in state_dict:
                    # Split bias similarly
                    in_proj_bias = state_dict.pop(in_proj_bias_key)
                    q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)

                    state_dict[f'{prefix}.q_proj.bias'] = q_bias
                    state_dict[f'{prefix}.k_proj.bias'] = k_bias
                    state_dict[f'{prefix}.v_proj.bias'] = v_bias

            # Clear optimizer state since parameter structure changed
            if 'optimizer_states' in checkpoint:
                checkpoint['optimizer_states'] = []
            if 'lr_schedulers' in checkpoint:
                checkpoint['lr_schedulers'] = []

            print("Migration complete! Training will resume with fresh optimizer state.")

    def sample_observations_from_images(self, images, num_obs=25):
        """
        Sample a fixed number of observations from images for visualization.
        Uses cosine-weighted latitude sampling to avoid over-sampling poles.
        """
        B, C, H, W = images.shape
        device = images.device

        observations = torch.zeros(B, num_obs, 8, device=device)
        obs_locations = torch.zeros(B, num_obs, 2, dtype=torch.long, device=device)

        # Cosine-weighted latitude sampling
        u = torch.rand(B, num_obs, device=device)
        lat_input = (2 * u - 1).clamp(-1.0, 1.0)  # Avoid arcsin NaN
        lat_degrees = torch.asin(lat_input) * (180 / math.pi)
        all_lats = ((lat_degrees + 90) / 180 * H).long().clamp(0, H - 1)

        # Uniform longitude sampling
        all_lons = torch.randint(0, W, (B, num_obs), device=device)

        # Store locations
        obs_locations[:, :, 0] = all_lats
        obs_locations[:, :, 1] = all_lons

        # Normalize locations to [-1, 1]
        observations[:, :, 0] = (all_lats.float() / H) * 2 - 1
        observations[:, :, 1] = (all_lons.float() / W) * 2 - 1

        # Gather observation values from images
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_obs)
        for ch in range(3):
            values = images[batch_indices, ch, all_lats, all_lons]
            observations[:, :, 2 + ch] = values
            # Full confidence for all channels in visualization
            observations[:, :, 5 + ch] = 1.0

        return observations, obs_locations

    def model_forward_with_obs_guidance(self, x, timesteps, class_labels, obs_embeds, obs_weights, obs_guidance_scale=1.0):
        """
        Forward pass with observation guidance (similar to classifier-free guidance).

        This method is robust to zero observation weights - all code paths handle the case
        where obs_weights is all zeros (skips cross-attention, returns gradient-supporting
        zero loss, etc).

        Args:
            x: (B, 4, 24, 48) noisy latents
            timesteps: (B,) timestep indices
            class_labels: (B, 256) continuous class embeddings
            obs_embeds: (B, N, 768) observation embeddings
            obs_weights: (B, N, 3) per-channel observation weights [0, 1]
            obs_guidance_scale: Observation guidance strength
                - 1.0: Normal (use observations as-is)
                - > 1.0: Stricter interpolation (amplify observation influence)
                - < 1.0: More global structure (reduce observation influence)
                - 0.0: Ignore observations entirely

        Returns:
            (B, 4, 24, 48) predicted noise/velocity
        """
        if obs_guidance_scale == 1.0:
            # No guidance, just normal forward pass
            return self.forward_model(x, timesteps, class_labels, obs_embeds, obs_weights)

        # Prediction with observations
        pred_with_obs = self.forward_model(x, timesteps, class_labels, obs_embeds, obs_weights)

        if obs_guidance_scale == 0.0:
            # No observations at all - use zero weights
            zero_weights = torch.zeros_like(obs_weights)
            return self.forward_model(x, timesteps, class_labels, obs_embeds, zero_weights)

        # Prediction without observations (zero weights)
        zero_weights = torch.zeros_like(obs_weights)
        pred_without_obs = self.forward_model(x, timesteps, class_labels, obs_embeds, zero_weights)

        # Apply observation guidance
        return pred_without_obs + obs_guidance_scale * (pred_with_obs - pred_without_obs)


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
        self.model = torch.compile(self.model)

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
    """VAE autoencoder training model (reconstruction loss, not diffusion)."""

    def __init__(self):
        super().__init__()
        self.vae = diffusers.models.AutoencoderTiny.from_pretrained("./taesd-iono-50k")
        self.vae = torch.compile(self.vae)

    def _shared_step(self, batch):
        """Shared VAE reconstruction step for train/val/test."""
        images = scale_to_diffusion(batch["images"])
        latents = self.vae.encode(images).latents
        decoded = self.vae.decode(latents).sample
        return F.mse_loss(decoded, images)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def forward(self, images):
        return self.vae.encode(scale_to_diffusion(images)).latents

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, eta_min=1e-6,
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
