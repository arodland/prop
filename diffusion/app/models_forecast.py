"""
Forecast model architecture for 24-hour autoregressive ionosphere prediction.

Key innovations:
1. Observation age encoding (9D observations with temporal age)
2. Previous map conditioning via input concatenation
3. Hierarchical CFG dropout for observations and previous map

Implementation notes:
- RopeEncoder(dim=64) for temporal encoding (produces 64-dim embeddings)
- CFG dropout sets first observation weight to 1e-8 (prevents all-True key_padding_mask NaN)
- Simplified temporal projection (Linear without LayerNorm for stability)
- Model uses torch.compile for performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import lightning as L
import diffusers
from diffusers.utils import numpy_to_pil
import math

# Import base components from existing models.py
from models import (
    DiffusionUtilsMixin,
    VAEUtilsMixin,
    RopeEncoder,
    TimestepEmbedder,
    FinalLayer,
    DiTBlockWithObservations,
    scale_from_diffusion,
)


class SpatioTemporalObservationEncoder(nn.Module):
    """
    Encodes observations with spatial location and temporal age.

    Input format: [lat, lon, fof2, mufd, hmf2, w_fof2, w_mufd, w_hmf2, age_hours]
    - lat, lon: Spatial coordinates in [-1, 1]
    - fof2, mufd, hmf2: Channel values (normalized)
    - w_fof2, w_mufd, w_hmf2: Per-channel confidence weights [0, 1]
    - age_hours: Observation age in hours (e.g., 1-47 hours old)

    Output: (B, N, 768) embeddings combining spatial, temporal, and channel information
    """

    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim

        # Spatial encoding: (lat, lon) → embed_dim // 4 (192)
        # Uses simple Linear projection (not RoPE) since spatial coordinates are Cartesian
        self.spatial_embed = nn.Linear(2, embed_dim // 4)

        # Temporal encoding: age_hours → embed_dim // 4 (192)
        # Use simple linear projection (most stable)
        self.temporal_proj = nn.Linear(1, embed_dim // 4)

        # Channel encoders: (value, weight) pairs for fof2, mufd, hmf2
        # Each gets embed_dim // 6 = 128 dimensions
        self.channel_encoders = nn.ModuleList([
            nn.Linear(2, embed_dim // 6) for _ in range(3)
        ])

        # Final projection: 192 (spatial) + 192 (temporal) + 384 (3×128 channels) = 768
        self.proj = nn.Linear(embed_dim // 2 + 3 * (embed_dim // 6), embed_dim)

    def forward(self, observations):
        """
        Args:
            observations: (B, N, 9) tensor of format:
                [lat, lon, fof2, mufd, hmf2, w_fof2, w_mufd, w_hmf2, age_hours]

        Returns:
            (B, N, embed_dim) observation embeddings
        """
        B, N, _ = observations.shape

        # Extract components
        spatial = observations[..., :2]        # (B, N, 2) - lat, lon
        values = observations[..., 2:5]        # (B, N, 3) - fof2, mufd, hmf2
        weights = observations[..., 5:8]       # (B, N, 3) - confidence weights
        age_hours = observations[..., 8:9]     # (B, N, 1) - observation age

        # Normalize age to [-1, 1] range
        # Observation ages typically span [1, 47] hours
        # Map [0, 48] → [-1, 1] for consistency with other normalized inputs
        age_normalized = age_hours / 48.0 * 2 - 1  # (B, N, 1)

        # Encode spatial (simple Linear projection)
        spatial_emb = self.spatial_embed(spatial)  # (B, N, embed_dim//4)

        # Encode temporal with MLP (simpler and more stable than high-freq RoPE)
        temporal_emb = self.temporal_proj(age_normalized)  # (B, N, embed_dim//4)

        # Encode channels with their confidence weights
        channel_embeds = []
        for i, encoder in enumerate(self.channel_encoders):
            ch_input = torch.stack([values[..., i], weights[..., i]], dim=-1)  # (B, N, 2)
            ch_emb = encoder(ch_input)  # (B, N, embed_dim//6)
            channel_embeds.append(ch_emb)

        # Concatenate all embeddings
        all_embeds = torch.cat([spatial_emb, temporal_emb] + channel_embeds, dim=-1)

        # Final projection to embed_dim
        output = self.proj(all_embeds)  # (B, N, embed_dim)

        return output


class ForecastObservationConditionedDiT(DiffusionUtilsMixin, VAEUtilsMixin, L.LightningModule):
    """
    DiT model for 24-hour autoregressive forecasting with:
    1. Aged observation conditioning (9D observations)
    2. Previous map conditioning via input concatenation
    3. Hierarchical CFG dropout

    Input: (B, 8, 24, 48) = concat([noisy_latents, prev_map_latents], dim=1)
    - Channels 0-3: Noisy current map (diffusion target)
    - Channels 4-7: Previous map (1 hour ago, deterministic)

    Observations: (B, N, 9) = [lat, lon, fof2, mufd, hmf2, w_fof2, w_mufd, w_hmf2, age_hours]
    """

    def __init__(
        self,
        input_size=(24, 48),
        patch_size=2,
        in_channels=8,  # 4 (noisy) + 4 (prev_map)
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        class_embed_size=256,
        max_observations=100,
        pred_type="v_prediction",
        cfg_dropout_obs=0.10,      # 10% null observations only
        cfg_dropout_prev=0.10,     # 10% null previous map only
        cfg_dropout_both=0.10,     # 10% null both (total 30% CFG)
    ):
        super().__init__()
        self.save_hyperparameters()

        # VAE (same as base ObservationConditionedDiT)
        self.vae = diffusers.models.AutoencoderTiny.from_pretrained("./taesd-iono-finetuned")
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        self.vae.latent_magnitude = 4.0

        # Parameter encoder (RoPE for global params: secular, toy, tod, ssn)
        self.param_encoder = RopeEncoder(dim=64)

        # NEW: Spatio-temporal observation encoder with age dimension
        self.obs_encoder = SpatioTemporalObservationEncoder(embed_dim=hidden_size)

        # Patchify (now handles 8 channels)
        self.input_h, self.input_w = input_size
        self.num_patches_h = self.input_h // patch_size
        self.num_patches_w = self.input_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.x_embedder = nn.Conv2d(
            in_channels=8,  # Concatenated [noisy, prev_map]
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
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

        # Final layer (outputs 4 channels, not 8 - only predicts current map)
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels=4)

        # Scheduler
        self.scheduler, self.inference_scheduler = self.create_diffusion_schedulers(pred_type)

        # Initialize weights
        self.initialize_weights()

        # Compile forward_model for speed (store separately to avoid checkpoint issues)
        self.model_forward = torch.compile(self.forward_model)

    def initialize_weights(self):
        """Initialize patch embedding and positional embeddings"""
        # Initialize patch embedding
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize observation encoder components with standard initialization
        nn.init.xavier_uniform_(self.obs_encoder.spatial_embed.weight)
        nn.init.zeros_(self.obs_encoder.spatial_embed.bias)

        nn.init.xavier_uniform_(self.obs_encoder.temporal_proj.weight)
        nn.init.zeros_(self.obs_encoder.temporal_proj.bias)

        for encoder in self.obs_encoder.channel_encoders:
            nn.init.xavier_uniform_(encoder.weight)
            nn.init.zeros_(encoder.bias)

        nn.init.xavier_uniform_(self.obs_encoder.proj.weight)
        nn.init.zeros_(self.obs_encoder.proj.bias)

    def apply_hierarchical_cfg_dropout(self, obs_embeds, obs_weights, prev_map_latents):
        """
        Apply hierarchical CFG dropout with three modes:
        - 10%: Null observations only
        - 10%: Null previous map only
        - 10%: Null both
        - 70%: Keep both (no dropout)

        Args:
            obs_embeds: (B, N, hidden_size) observation embeddings
            obs_weights: (B, N, 3) observation confidence weights
            prev_map_latents: (B, 4, 24, 48) previous map latents

        Returns:
            Tuple of (obs_embeds, obs_weights, prev_map_latents) with dropout applied
        """
        batch_size = obs_embeds.shape[0]
        device = obs_embeds.device

        dropout_type = torch.rand(batch_size, device=device)

        # 10%: null both
        null_both_mask = dropout_type < self.hparams.cfg_dropout_both

        # 10%: null observations only
        null_obs_mask = (dropout_type >= self.hparams.cfg_dropout_both) & \
                        (dropout_type < self.hparams.cfg_dropout_both + self.hparams.cfg_dropout_obs)

        # 10%: null previous map only
        null_prev_mask = (dropout_type >= self.hparams.cfg_dropout_both + self.hparams.cfg_dropout_obs) & \
                         (dropout_type < self.hparams.cfg_dropout_both + self.hparams.cfg_dropout_obs + self.hparams.cfg_dropout_prev)

        # Combined masks
        null_obs_total = null_both_mask | null_obs_mask
        null_prev_total = null_both_mask | null_prev_mask

        # Apply observation dropout
        if null_obs_total.any():
            obs_embeds = obs_embeds.clone()
            obs_weights = obs_weights.clone()
            obs_embeds[null_obs_total] = 0.0
            obs_weights[null_obs_total] = 0.0

            # BUGFIX: For batch elements with all-zero weights, set first obs weight to tiny epsilon
            # This prevents all-True key_padding_mask in cross-attention (which causes NaN in softmax)
            # Since embeddings are zero, this has minimal impact on output
            obs_weights[null_obs_total, 0, :] = 1e-8

        # Apply previous map dropout
        if null_prev_total.any():
            prev_map_latents = prev_map_latents.clone()
            prev_map_latents[null_prev_total] = 0.0

        return obs_embeds, obs_weights, prev_map_latents

    def forward_model(self, x, timesteps, class_labels, obs_embeds=None, obs_weights=None):
        """
        Forward pass through the DiT model.

        Args:
            x: (B, 8, 24, 48) concatenated [noisy_latents, prev_map_latents]
            timesteps: (B,) timestep indices
            class_labels: (B, 256) continuous class embeddings (from RoPE)
            obs_embeds: (B, N, 768) observation embeddings (optional)
            obs_weights: (B, N, 3) per-channel observation weights [0, 1] (optional)

        Returns:
            (B, 4, 24, 48) predicted noise/velocity for current map only
        """
        batch_size = x.shape[0]

        # Handle scalar timesteps
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.shape[0] != batch_size:
            timesteps = timesteps.expand(batch_size)

        # Patchify 8-channel input and add position embeddings
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

        # Final layer (outputs 4 channels)
        x = self.final_layer(x, c, self.num_patches_h, self.num_patches_w)  # (B, 4, H, W)

        return x

    def model_loss(self, prediction, x, noise, step):
        """Compute diffusion loss using shared helper from DiffusionUtilsMixin"""
        return self.compute_standard_diffusion_loss(
            prediction, x, noise, step, self.hparams.pred_type
        )

    def sample_observations_with_age(self, images_sequence, forecast_timestep, num_obs=50):
        """
        Sample observations from historical window with age information.

        Args:
            images_sequence: (B, num_timesteps, C, H, W) sequence of maps
            forecast_timestep: Target forecast timestep index (e.g., 25 for first forecast hour)
            num_obs: Number of observations to sample per batch element

        Returns:
            observations: (B, N, 9) tensor with age as 9th dimension
        """
        B, T, C, H, W = images_sequence.shape
        device = images_sequence.device

        observations = torch.zeros(B, num_obs, 9, device=device)

        for b in range(B):
            for n in range(num_obs):
                # Sample age uniformly from [forecast_timestep-24, forecast_timestep]
                # This gives us observations from the 24-hour historical window
                min_age = max(1, forecast_timestep - 24)  # At least 1 hour old
                max_age = forecast_timestep
                age_hours = torch.rand(1, device=device).item() * (max_age - min_age) + min_age

                # Map age to historical timestep
                obs_timestep = int(forecast_timestep - age_hours)
                obs_timestep = max(0, min(T - 1, obs_timestep))

                # Sample spatial location (cosine-weighted latitude)
                u = torch.rand(1, device=device).item()
                lat_degrees = math.asin(2 * u - 1) * (180 / math.pi)  # [-90, 90]
                lat_pixel = int((lat_degrees + 90) / 180 * H)
                lat_pixel = max(0, min(H - 1, lat_pixel))

                lon_pixel = torch.randint(0, W, (1,), device=device).item()

                # Get values from historical map
                fof2 = images_sequence[b, obs_timestep, 0, lat_pixel, lon_pixel].item()
                mufd = images_sequence[b, obs_timestep, 1, lat_pixel, lon_pixel].item()
                hmf2 = images_sequence[b, obs_timestep, 2, lat_pixel, lon_pixel].item()

                # Normalize location to [-1, 1]
                lat_norm = (lat_pixel / H) * 2 - 1
                lon_norm = (lon_pixel / W) * 2 - 1

                # Sample confidence (Beta distribution biased toward high confidence)
                confidence = torch.distributions.Beta(2.0, 1.0).sample().item()

                # Build observation: [lat, lon, fof2, mufd, hmf2, w_fof2, w_mufd, w_hmf2, age_hours]
                observations[b, n] = torch.tensor([
                    lat_norm, lon_norm,
                    fof2, mufd, hmf2,
                    confidence, confidence, confidence,
                    age_hours
                ], device=device)

        return observations

    def training_step(self, batch, batch_idx):
        """
        Training step for forecast model.

        Batch format:
            - image_t: (B, C, H, W) target map at time t
            - image_t_minus_1: (B, C, H, W) previous map at time t-1
            - params: (B, 4) global parameters [secular, toy, tod, ssn]
            - observations: (B, N, 9) observations with age
        """
        image_t = batch["image_t"]
        image_t_minus_1 = batch["image_t_minus_1"]
        params = batch["params"]
        observations = batch["observations"]

        # SSN corruption (30% of batches) - forces model to use observations and previous map
        corrupted_params = params.clone()
        corruption_type = torch.rand(1).item()

        if corruption_type < 0.15:  # 15%: Zero out SSN
            corrupted_params[:, 3] = 0.0
        elif corruption_type < 0.30:  # 15%: Add noise to SSN
            ssn_noise = torch.randn_like(corrupted_params[:, 3]) * 0.3
            corrupted_params[:, 3] = corrupted_params[:, 3] + ssn_noise

        # Encode images to latents
        latents_t = self.encode_images_to_latents(image_t)
        latents_t_minus_1 = self.encode_images_to_latents(image_t_minus_1)

        # Encode parameters and observations (use corrupted params)
        param_embeds = self.param_encoder.rope_enc(corrupted_params)
        obs_embeds = self.obs_encoder(observations)
        obs_weights = observations[..., 5:8]  # Per-channel weights

        # Apply hierarchical CFG dropout
        obs_embeds, obs_weights, latents_t_minus_1 = self.apply_hierarchical_cfg_dropout(
            obs_embeds, obs_weights, latents_t_minus_1
        )

        # Diffusion forward process on current map
        noise = torch.randn_like(latents_t)
        steps = torch.randint(
            self.scheduler.config.num_train_timesteps,
            (image_t.size(0),),
            device=self.device
        )
        noisy_latents = self.scheduler.add_noise(latents_t, noise, steps)

        # Concatenate noisy current map with previous map
        x_input = torch.cat([noisy_latents, latents_t_minus_1], dim=1)  # (B, 8, 24, 48)

        # Model prediction
        model_pred = self.model_forward(x_input, steps, param_embeds, obs_embeds, obs_weights)

        # Diffusion loss
        loss = self.model_loss(model_pred, latents_t, noise, steps)

        # Get current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']

        self.log_dict({
            "train_loss": loss,
            "lr": current_lr,
        }, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        image_t = batch["image_t"]
        image_t_minus_1 = batch["image_t_minus_1"]
        params = batch["params"]
        observations = batch["observations"]

        # Encode
        latents_t = self.encode_images_to_latents(image_t)
        latents_t_minus_1 = self.encode_images_to_latents(image_t_minus_1)

        param_embeds = self.param_encoder.rope_enc(params)
        obs_embeds = self.obs_encoder(observations)
        obs_weights = observations[..., 5:8]

        # Diffusion
        noise = torch.randn_like(latents_t)
        steps = torch.randint(
            self.scheduler.config.num_train_timesteps,
            (image_t.size(0),),
            device=self.device
        )
        noisy_latents = self.scheduler.add_noise(latents_t, noise, steps)

        x_input = torch.cat([noisy_latents, latents_t_minus_1], dim=1)

        model_pred = self.model_forward(x_input, steps, param_embeds, obs_embeds, obs_weights)

        loss = self.model_loss(model_pred, latents_t, noise, steps)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_save_checkpoint(self, checkpoint):
        """Generate sample forecast images when saving checkpoint"""
        with torch.no_grad():
            # Sample a single validation example
            val_dataloader = self.trainer.datamodule.val_dataloader()
            val_dataset = val_dataloader.dataset

            # Get one random sample
            idx = torch.randint(len(val_dataset), (1,)).item()
            sample = val_dataset[idx]

            # Move to device and add batch dimension
            image_t = sample["image_t"].unsqueeze(0).to(self.device)
            image_t_minus_1 = sample["image_t_minus_1"].unsqueeze(0).to(self.device)
            params = sample["params"].unsqueeze(0).to(self.device)
            observations = sample["observations"].unsqueeze(0).to(self.device)

            # Encode
            latents_t_minus_1 = self.encode_images_to_latents(image_t_minus_1)
            param_embeds = self.param_encoder.rope_enc(params)
            obs_embeds = self.obs_encoder(observations)
            obs_weights = observations[..., 5:8]

            # Generate with CFG
            generated_latents = self.generate_with_cfg(
                latents_t_minus_1,
                observations,
                params,
                cfg_scale_obs=1.5,
                cfg_scale_prev=2.0,
                cfg_scale_joint=1.0,
                num_inference_steps=50,
            )

            # Crop latents to original size before decoding
            generated_latents_cropped = generated_latents[..., :23, :46]
            generated_images = self.decode_latents_to_images(generated_latents_cropped)

            # Ground truth
            gt_latents = self.encode_images_to_latents(image_t)
            gt_latents_cropped = gt_latents[..., :23, :46]
            gt_images = self.decode_latents_to_images(gt_latents_cropped)

            # Stack: [prev, generated, ground_truth]
            images_to_save = torch.cat([
                image_t_minus_1,  # Previous map
                generated_images,  # Generated forecast
                gt_images,  # Ground truth
            ], dim=0)

            # Visualize only first 3 channels (foF2, MUFD, hmF2)
            pil_images = numpy_to_pil(
                images_to_save[:, :3, ...].cpu().permute(0, 2, 3, 1).detach().numpy())

            # Save as grid
            tv.utils.save_image(
                tv.utils.make_grid(
                    torch.stack([tv.transforms.functional.to_tensor(img) for img in pil_images]),
                    nrow=3,
                ),
                "out/forecast_checkpoint.png",
            )

    def on_load_checkpoint(self, checkpoint):
        """Migrate old nn.MultiheadAttention weights to new MultiheadSDPA format"""
        state_dict = checkpoint['state_dict']

        # Check if we need migration (old format has in_proj_weight)
        needs_migration = any('cross_attn.in_proj_weight' in k for k in state_dict.keys())

        if needs_migration:
            print("Migrating checkpoint from nn.MultiheadAttention to MultiheadSDPA format...")
            for i in range(len(self.blocks)):
                prefix = f'blocks.{i}.cross_attn'

                # Check if old keys exist
                in_proj_weight_key = f'{prefix}.in_proj_weight'
                in_proj_bias_key = f'{prefix}.in_proj_bias'
                out_proj_weight_key = f'{prefix}.out_proj.weight'
                out_proj_bias_key = f'{prefix}.out_proj.bias'

                if in_proj_weight_key in state_dict:
                    # Split in_proj_weight into q, k, v projections
                    in_proj_weight = state_dict[in_proj_weight_key]
                    embed_dim = in_proj_weight.shape[1]

                    # in_proj_weight is (3*embed_dim, embed_dim) containing [Q; K; V] stacked
                    q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)

                    state_dict[f'{prefix}.q_proj.weight'] = q_weight
                    state_dict[f'{prefix}.k_proj.weight'] = k_weight
                    state_dict[f'{prefix}.v_proj.weight'] = v_weight

                    # Split in_proj_bias
                    if in_proj_bias_key in state_dict:
                        in_proj_bias = state_dict[in_proj_bias_key]
                        q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)

                        state_dict[f'{prefix}.q_proj.bias'] = q_bias
                        state_dict[f'{prefix}.k_proj.bias'] = k_bias
                        state_dict[f'{prefix}.v_proj.bias'] = v_bias

                        del state_dict[in_proj_bias_key]

                    # out_proj weights are already in correct format (no change needed)
                    # Just remove old combined weight
                    del state_dict[in_proj_weight_key]

            # Clear optimizer state since parameter structure changed
            if 'optimizer_states' in checkpoint:
                print("Clearing optimizer state due to architecture change...")
                checkpoint['optimizer_states'] = []
            if 'lr_schedulers' in checkpoint:
                checkpoint['lr_schedulers'] = []

            print("Migration complete! Training will resume with fresh optimizer state.")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        # Exponential decay from 1e-4 to 1e-5 over 200 epochs
        # gamma = 10^(-1/200) ≈ 0.98855
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98855)
        return [optimizer], [scheduler]

    def generate_with_cfg(
        self,
        prev_map_latents,
        observations,
        params,
        cfg_scale_obs=1.5,
        cfg_scale_prev=2.0,
        cfg_scale_joint=1.0,
        num_inference_steps=50,
    ):
        """
        Generate with hierarchical CFG guidance.

        Args:
            prev_map_latents: (B, 4, 24, 48) previous map latents
            observations: (B, N, 9) observations with age
            params: (B, 4) global parameters
            cfg_scale_obs: Guidance scale for observations
            cfg_scale_prev: Guidance scale for previous map
            cfg_scale_joint: Guidance scale for joint conditioning
            num_inference_steps: Number of denoising steps

        Returns:
            (B, 4, 24, 48) predicted latents
        """
        batch_size = prev_map_latents.shape[0]
        device = prev_map_latents.device

        # Encode conditioning
        param_embeds = self.param_encoder.rope_enc(params)
        obs_embeds = self.obs_encoder(observations)
        obs_weights = observations[..., 5:8]

        # Initialize noise
        x = torch.randn(batch_size, 4, 24, 48, device=device)

        # Set up scheduler
        self.inference_scheduler.set_timesteps(num_inference_steps, device=device)

        for t in self.inference_scheduler.timesteps:
            # Expand timestep for batch
            t_batch = t.expand(batch_size)

            # Four forward passes for hierarchical CFG
            # 1. Both conditioning
            x_input = torch.cat([x, prev_map_latents], dim=1)
            pred_both = self.model_forward(x_input, t_batch, param_embeds, obs_embeds, obs_weights)

            # 2. No observations
            pred_no_obs = self.model_forward(
                x_input, t_batch, param_embeds,
                obs_embeds * 0, obs_weights * 0
            )

            # 3. No previous map
            x_input_no_prev = torch.cat([x, prev_map_latents * 0], dim=1)
            pred_no_prev = self.model_forward(
                x_input_no_prev, t_batch, param_embeds,
                obs_embeds, obs_weights
            )

            # 4. Null (no conditioning)
            pred_null = self.model_forward(
                x_input_no_prev, t_batch, param_embeds,
                obs_embeds * 0, obs_weights * 0
            )

            # Hierarchical CFG combination
            pred = pred_null + \
                   cfg_scale_obs * (pred_no_prev - pred_null) + \
                   cfg_scale_prev * (pred_no_obs - pred_null) + \
                   cfg_scale_joint * (pred_both - pred_no_obs - pred_no_prev + pred_null)

            # Denoise step
            step_output = self.inference_scheduler.step(pred, t, x)
            x = step_output.prev_sample

        return step_output.pred_original_sample
