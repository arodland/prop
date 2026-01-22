from flask import Flask, request, make_response
from data import json, jsonapi, hdf5
import os
import io
import h5py
import hdf5plugin
import datetime
import psycopg
import math
import torch
import torchvision as tv
import diffusers
from models import DiffusionModel, ConditionedDiffusionModel, DiTDiffusionModel, ObservationConditionedDiT, GuidanceModel
import torch.nn.functional as F
from util import scale_from_diffusion

model_params = {
    "unet_cfg": {
        "mode": "cfg",
        "diffusion_checkpoint": "/checkpoints/diffusion/cdiffusion-v-epoch=99-val_loss=0.0019.ckpt",
        "steps": 100,
        "guidance_scale": 5,
        "fit_scale_start": 20.0,
        "fit_scale_end": 100.0,
        "ema_alpha": 0.7,
        "max_grad": 10.0,
    },
    "unet_repaint": {
        "mode": "repaint",
        "diffusion_checkpoint": "/checkpoints/diffusion/cdiffusion-v-epoch=99-val_loss=0.0019.ckpt",
        "steps": 100,
        "guidance_scale": 5,
        "jump_length": 10,
        "jump_n_sample": 10,
    },
    "dit_cfg": {
        "mode": "cfg",
        "model_class": "DiTDiffusionModel",
        "diffusion_checkpoint": "/checkpoints/diffusion/dit-diffusion-v-epoch=226-val_loss=0.001.ckpt",
        "steps": 100,
        "guidance_scale": 5,
        "fit_scale_start": 20.0,
        "fit_scale_end": 100.0,
        "ema_alpha": 0.7,
        "max_grad": 10.0,
    },
    "dit_repaint": {
        "mode": "repaint",
        "model_class": "DiTDiffusionModel",
        "diffusion_checkpoint": "/checkpoints/diffusion/dit-diffusion-v-epoch=226-val_loss=0.001.ckpt",
        "steps": 100,
        "guidance_scale": 5,
        "jump_length": 10,
        "jump_n_sample": 10,
    },
    "dit_obs": {
        "mode": "observation_conditioned",
        "model_class": "ObservationConditionedDiT",
        "diffusion_checkpoint": "/checkpoints/diffusion/obs-dit-v-epoch=211-val_loss=0.0023.ckpt",
        "steps": 25,
        "guidance_scale": 1.2,
        "obs_guidance_scale": 1.0,
    },
}

metrics = {
    "fof2": {"min": 1.5, "max": 15.0, "ch": 0},
    "mufd": {"min": 5.0, "max": 45.0, "ch": 1},
    "hmf2": {"min": 150.0, "max": 450.0, "ch": 2},
}

geometry_scale = 45.0
max_dilate = 44


def get_current():
    return jsonapi.get_data("http://localhost:%s/stations.json" % os.getenv("API_PORT"))


def get_pred(run_id, ts):
    return jsonapi.get_data(
        "http://localhost:%s/pred.json?run_id=%d&ts=%d"
        % (os.getenv("API_PORT"), run_id, ts)
    )


def get_holdouts(run_id):
    return json.get_data(
        "http://localhost:%s/holdout?run_id=%d" % (os.getenv("API_PORT"), run_id)
    )


def get_irimap(run_id, ts):
    return hdf5.get_data(
        "http://localhost:%s/irimap.h5?run_id=%d&ts=%d"
        % (os.getenv("API_PORT"), run_id, ts)
    )


def filter_holdouts(df, holdouts):
    if len(holdouts):
        holdout_station_ids = [row["station"]["id"] for row in holdouts]
        for ii in holdout_station_ids:
            df = df.drop(df[df["station.id"] == ii].index)

    return df


def lat_lon_distance(lat1, lon1, lat2, lon2):
    # Convert to radians
    lat1_rad = torch.deg2rad(
        torch.as_tensor(lat1) - 90.0
    )  # Adjust latitude to be from -90 to 90
    lon1_rad = torch.deg2rad(
        torch.as_tensor(lon1) - 180.0
    )  # Adjust longitude to be from -180 to 180
    lat2_rad = torch.deg2rad(
        torch.as_tensor(lat2) - 90.0
    )  # Adjust latitude to be from -90 to 90
    lon2_rad = torch.deg2rad(
        torch.as_tensor(lon2) - 180.0
    )  # Adjust longitude to be from -180 to 180

    # Spherical law of cosines
    cos_angle = torch.sin(lat1_rad) * torch.sin(lat2_rad) + torch.cos(
        lat1_rad
    ) * torch.cos(lat2_rad) * torch.cos(lon2_rad - lon1_rad)

    cos_angle = cos_angle.clamp(
        -1.0, 1.0
    )  # Ensure the value is within the valid range for acos

    # Return angular distance in degrees
    return torch.rad2deg(torch.acos(cos_angle))


def dilations(lats, lons, lat, lon, dilate_by):
    """Dilates the mask by a specified number of pixels."""
    dilated_masks = []

    distance = lat_lon_distance(lats, lons, lat, lon)

    for i in range(dilate_by + 1):
        mask = (i + 0.5 - distance).clip(0.0, 1.0)
        dilated_masks.append(mask)
    return dilated_masks


def create_targets(df_pred, num_samples, device):
    out_targets = [
        torch.zeros((3, 184, 368), device=device) for _ in range(max_dilate + 1)
    ]
    dilated_masks = [
        torch.zeros((3, 184, 368), device=device) for _ in range(max_dilate + 1)
    ]
    unweighted_masks = [
        torch.zeros((3, 184, 368), device=device) for _ in range(max_dilate + 1)
    ]

    lats, lons = torch.meshgrid(
        torch.arange(0, 184, dtype=torch.float32),
        torch.arange(0, 368, dtype=torch.float32),
        indexing="ij",
    )
    lats = lats.to(device)
    lons = lons.to(device)

    # Draw the known data for each station into the masks, at the 45 different dilation scales
    for _, station in df_pred.iterrows():
        lat = station["station.latitude"] + 90
        lon = station["station.longitude"] + 180
        cs = float(station["cs"])

        pos_masks = dilations(lats, lons, lat, lon, max_dilate)

        for metric in metrics:
            if station[metric] is None:
                continue
            ch = metrics[metric]["ch"]
            min_val, max_val = metrics[metric]["min"], metrics[metric]["max"]
            normalized_value = (station[metric] - min_val) / (max_val - min_val)
            for i, m in enumerate(pos_masks):
                out_targets[i][ch, ...] += m * cs * normalized_value
                dilated_masks[i][ch, ...] += m * cs
                unweighted_masks[i][ch, ...] += m

    # Normalize the values so that out_targets is an average of metrics (weighted by station confidence)
    # and dilated masks is an average of confidence scores.
    for i in range(max_dilate + 1):
        out_targets[i] /= torch.clip(dilated_masks[i], 1e-6, None)
        out_targets[i] = torch.clip(out_targets[i], 0.0, 1.0)
        out_targets[i] = out_targets[i].expand(num_samples, -1, -1, -1)

        dilated_masks[i] /= torch.clip(unweighted_masks[i], 1e-6, None)
        dilated_masks[i] = torch.clip(dilated_masks[i], 0.0, 1.0)
        dilated_masks[i] = dilated_masks[i].expand(num_samples, -1, -1, -1)

    return out_targets, dilated_masks


def make_guidance_target(ts, essn):
    dt = datetime.datetime.fromtimestamp(ts)
    year = dt.year
    toy = (
        dt - dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    ) / datetime.timedelta(days=365)
    toy = min(max(toy, 0.0), 1.0)
    tod = (
        dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)
    ) / datetime.timedelta(hours=24)

    return torch.tensor(
        [
            (year - 2000.0 + toy) / 50.0,
            math.sin(toy * 2 * math.pi),
            math.cos(toy * 2 * math.pi),
            math.sin(tod * 2 * math.pi),
            math.cos(tod * 2 * math.pi),
            essn / 100.0 - 1.0,
        ]
    )


def make_cfg_target(ts, essn):
    dt = datetime.datetime.fromtimestamp(ts)
    year = dt.year
    toy = (
        dt - dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    ) / datetime.timedelta(days=365)
    toy = min(max(toy, 0.0), 1.0)
    tod = (
        dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)
    ) / datetime.timedelta(hours=24)

    return torch.tensor(
        [
            (year - 2000.0 + toy) / 50.0,
            toy,
            tod,
            essn / 100.0 - 1.0,
        ]
    )


def run_diffusion(model, ts, essn, df_pred, num_samples=5):
    params = model_params[model]
    cfg = params["mode"] == "cfg"

    if cfg:
        if "dm" not in params:
            # Check if a specific model class is specified (e.g., DiTDiffusionModel)
            model_class_name = params.get("model_class", "ConditionedDiffusionModel")
            if model_class_name == "DiTDiffusionModel":
                params["dm"] = DiTDiffusionModel.load_from_checkpoint(
                    params["diffusion_checkpoint"]
                ).to(device="cuda")
            else:
                params["dm"] = ConditionedDiffusionModel.load_from_checkpoint(
                    params["diffusion_checkpoint"]
                ).to(device="cuda")
            params["dm"].eval()
        dm = params["dm"]
    else:
        if "dm" not in params:
            params["dm"] = DiffusionModel.load_from_checkpoint(
                params["diffusion_checkpoint"]
            ).to(device="cuda")
            params["dm"].eval()
        dm = params["dm"]
        if "gm" not in params:
            params["gm"] = GuidanceModel.load_from_checkpoint(
                params["guidance_checkpoint"]
            ).to(device="cuda")
            params["gm"].eval()
        gm = params["gm"]

    scheduler = diffusers.schedulers.DDPMScheduler(rescale_betas_zero_snr=True)
    scheduler.set_timesteps(params["steps"], device=dm.device)

    targets, masks = create_targets(df_pred, device=dm.device, num_samples=num_samples)

    if cfg:
        guidance_target = (
            make_cfg_target(ts, essn).to(device=dm.device).expand(num_samples, -1)
        )
        # Use rope_enc for DiT (matches training), full encoder for UNet
        if isinstance(dm, DiTDiffusionModel):
            encoded_target = dm.param_encoder.rope_enc(guidance_target)
        else:
            encoded_target = dm.param_encoder(guidance_target)
        null_target = torch.zeros_like(encoded_target)
    else:
        guidance_target = (
            make_guidance_target(ts, essn).to(device=dm.device).expand(num_samples, -1)
        )

    x = torch.randn((num_samples, 4, 24, 48), device=dm.device)

    zero_dilate_timestep = params["steps"] if cfg else params["steps"] * 0.95
    grad_ema = None

    for i, t in enumerate(scheduler.timesteps):
        torch.compiler.cudagraph_mark_step_begin()
        model_input = scheduler.scale_model_input(x, t)

        # Make a denoising step, with the appropriate sort of guidance
        if cfg:
            with torch.no_grad():
                noise_pred_guided = dm.model(
                    model_input, t, class_labels=encoded_target
                ).sample
                noise_pred_unguided = dm.model(
                    model_input, t, class_labels=null_target
                ).sample
                noise_pred = noise_pred_unguided + params["guidance_scale"] * (
                    noise_pred_guided - noise_pred_unguided
                )
        else:
            with torch.no_grad():
                noise_pred = dm.model(model_input, t).sample

        # Get a predicted x0 and VAE-decode it.
        x = x.detach().requires_grad_()
        x0 = scheduler.step(noise_pred, t, x).pred_original_sample
        x0_decoded = scale_from_diffusion(dm.vae.decode(dm.scale_latents(x0)).sample)

        # Cut down to 181x361 valid data, then re-pad to 184x368 for the sake of the guidance model
        x0_decoded = x0_decoded[..., :181, :361]
        x0_decoded = F.pad(x0_decoded, (0, 7, 0, 3))

        # Calculate loss for the geometry not being right
        wrap_loss_lat = F.mse_loss(
            x0_decoded[..., 0],
            x0_decoded[..., 360],
        )
        wrap_loss_lon = torch.var(x0_decoded[..., 0, :]) + torch.var(
            x0_decoded[..., 180, :]
        )
        loss = geometry_scale * (wrap_loss_lat + wrap_loss_lon)

        # Calculate guidance loss if doing classifier guidance
        if not cfg:
            guidance_out = gm.model(x0_decoded)
            guidance_loss = F.mse_loss(guidance_out, guidance_target)
            loss += guidance_loss * params["guidance_scale"]

        # Fetch the fit target/mask for the current timestep
        dilate_mask_by = int(
            round(((zero_dilate_timestep - i) * max_dilate) / zero_dilate_timestep)
        )
        dilate_mask_by = 0 if dilate_mask_by < 0 else dilate_mask_by
        mask_dilated = masks[dilate_mask_by]
        out_target = targets[dilate_mask_by]

        # And apply the target-fitting loss
        if cfg or i <= zero_dilate_timestep:
            fit_loss = (x0_decoded - out_target).pow(2).mul(
                mask_dilated
            ).sum() / mask_dilated.sum()
            progress = i / params["steps"]
            fit_scale = (
                params["fit_scale_start"]
                + (params["fit_scale_end"] - params["fit_scale_start"]) * progress
            )
            loss += fit_loss * fit_scale

        print(i, loss)

        # Backward the loss and take a step to decrease it
        grad = -torch.autograd.grad(loss, x)[0]
        grad = torch.clamp(grad, -params["max_grad"], params["max_grad"])
        if grad_ema is None or params["ema_alpha"] == 0.0:
            grad_ema = grad
        else:
            grad_ema = (
                params["ema_alpha"] * grad_ema + (1.0 - params["ema_alpha"]) * grad
            )
        x = x.detach() + grad_ema
        x = scheduler.step(noise_pred, t, x).prev_sample

    outs = scale_from_diffusion(dm.vae.decode(dm.scale_latents(x)).sample)
    outs = outs[..., :181, :361]
    # Force 180E and 180W to be equal
    outs[..., 360] = outs[..., 0]

    ensemble = torch.quantile(outs, 0.5, dim=0)

    ret = {}
    for metric in metrics:
        mval = ensemble[metrics[metric]["ch"], ...].detach().cpu().numpy()
        mval = (
            mval * (metrics[metric]["max"] - metrics[metric]["min"])
            + metrics[metric]["min"]
        )
        ret[metric] = mval

    ret["md"] = ret["mufd"] / ret["fof2"]
    return ret


def run_repaint(model, ts, essn, df_pred, num_samples=5):
    """Run RePaint inpainting-based diffusion inference.

    Uses the RePaint algorithm to directly replace masked regions at each timestep
    instead of using gradient-based guidance.
    """
    params = model_params[model]

    # Load the model
    if "dm" not in params:
        # Check if a specific model class is specified (e.g., DiTDiffusionModel)
        model_class_name = params.get("model_class", "ConditionedDiffusionModel")
        if model_class_name == "DiTDiffusionModel":
            params["dm"] = DiTDiffusionModel.load_from_checkpoint(
                params["diffusion_checkpoint"]
            ).to(device="cuda")
        else:
            params["dm"] = ConditionedDiffusionModel.load_from_checkpoint(
                params["diffusion_checkpoint"]
            ).to(device="cuda")
        params["dm"].eval()
    dm = params["dm"]

    scheduler = diffusers.schedulers.DDPMScheduler(
        rescale_betas_zero_snr=True,
        prediction_type=dm.inference_scheduler.config.prediction_type,
    )
    scheduler.set_timesteps(params["steps"], device=dm.device)

    # Create targets and masks at all dilation levels
    targets, masks = create_targets(df_pred, device=dm.device, num_samples=num_samples)

    # Create guidance target for CFG
    guidance_target = (
        make_cfg_target(ts, essn).to(device=dm.device).expand(num_samples, -1)
    )
    # Use rope_enc for DiT (matches training), full encoder for UNet
    if isinstance(dm, DiTDiffusionModel):
        encoded_target = dm.param_encoder.rope_enc(guidance_target)
    else:
        encoded_target = dm.param_encoder(guidance_target)
    null_target = torch.zeros_like(encoded_target)

    # Start from random noise
    x = torch.randn((num_samples, 4, 24, 48), device=dm.device)

    # Encode the known data (targets) to latent space for inpainting
    known_latents = []
    for i in range(max_dilate + 1):
        target_data = targets[i][0:1, ...]  # Take first sample, shape: (1, 3, 184, 368)
        target_data = (target_data * 2.0) - 1.0  # Scale to [-1, 1] for VAE
        with torch.no_grad():
            target_latent = dm.vae.encode(target_data).latents
            target_latent = dm.unscale_latents(target_latent)
            # Pad to 24x48 to match diffusion model latent size
            target_latent = F.pad(
                target_latent,
                (0, 48 - target_latent.shape[-1], 0, 24 - target_latent.shape[-2]),
            )
        known_latents.append(target_latent.expand(num_samples, -1, -1, -1))

    # Downsample masks to latent resolution
    latent_masks = []
    for i in range(max_dilate + 1):
        mask_full = masks[i][0:1, :, :, :]  # Shape: (1, 3, 184, 368)
        latent_mask = F.avg_pool2d(
            mask_full, kernel_size=8, stride=8
        )  # Shape: (1, 3, 23, 46)
        latent_mask = F.pad(
            latent_mask, (0, 48 - latent_mask.shape[-1], 0, 24 - latent_mask.shape[-2])
        )  # (1, 3, 24, 48)
        latent_masks.append(
            latent_mask[:, 0:1, :, :].expand(num_samples, 1, -1, -1)
        )  # (num_samples, 1, 24, 48)

    # RePaint loop with resampling
    jump_length = params.get("jump_length", 10)
    jump_n_sample = params.get("jump_n_sample", 10)
    mask_strength = params.get("mask_strength", 1.0)

    i = 0  # Current index in scheduler.timesteps
    jump_timesteps = set(range(0, params["steps"] - jump_length, jump_length))
    jump_counts = {j: jump_n_sample - 1 for j in jump_timesteps}

    while i < params["steps"]:
        t = scheduler.timesteps[i]

        # Determine which mask to use based on progress
        progress = 1.0 - (i / params["steps"])
        dilate_mask_by = int(round(progress * max_dilate))
        dilate_mask_by = min(max(dilate_mask_by, 0), max_dilate)

        mask_latent = latent_masks[dilate_mask_by]
        known_latent = known_latents[dilate_mask_by]

        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()

            # Sample x_known_{t-1} from the known region
            # We're denoising FROM timestep t TO timestep t-1
            # So we need to sample x_known at the TARGET noise level (t-1)
            # x_known_{t-1} = sqrt(alpha_bar_{t-1}) * x_0 + sqrt(1 - alpha_bar_{t-1}) * epsilon
            if i < params["steps"] - 1:
                # Get the next timestep (lower noise level)
                t_next = scheduler.timesteps[i + 1]
                alpha_bar = scheduler.alphas_cumprod[t_next.long()]
                noise = torch.randn_like(known_latent)
                x_known = (
                    torch.sqrt(alpha_bar) * known_latent
                    + torch.sqrt(1 - alpha_bar) * noise
                )
            else:
                # At final timestep (i == params["steps"] - 1), target is clean x_0
                x_known = known_latent

            # Denoise the unknown region with CFG
            model_input = scheduler.scale_model_input(x, t)
            noise_pred_guided = dm.model(
                model_input, t, class_labels=encoded_target
            ).sample
            noise_pred_unguided = dm.model(
                model_input, t, class_labels=null_target
            ).sample
            noise_pred = noise_pred_unguided + params["guidance_scale"] * (
                noise_pred_guided - noise_pred_unguided
            )
            x_unknown = scheduler.step(noise_pred, t, x).prev_sample

            # Combine known and unknown regions
            # Apply mask_strength to control adherence to target points
            effective_mask = mask_latent * mask_strength
            x = effective_mask * x_known + (1 - effective_mask) * x_unknown

        # Check if we should resample (jump back) at this timestep
        if i in jump_counts and jump_counts[i] > 0 and i < params["steps"] - 1:
            jump_counts[i] -= 1
            # Add noise to jump forward in diffusion time
            with torch.no_grad():
                for _ in range(jump_length):
                    if i < params["steps"] - 1:
                        t_fwd = scheduler.timesteps[min(i + 1, params["steps"] - 1)]
                        beta = scheduler.betas[t_fwd.long()]
                        noise = torch.randn_like(x)
                        x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise
        else:
            # Move to next timestep
            i += 1

    # Decode final result
    outs = scale_from_diffusion(dm.vae.decode(dm.scale_latents(x)).sample)
    outs = outs[..., :181, :361]
    # Force 180E and 180W to be equal
    outs[..., 360] = outs[..., 0]

    ensemble = torch.quantile(outs, 0.5, dim=0)

    ret = {}
    for metric in metrics:
        mval = ensemble[metrics[metric]["ch"], ...].detach().cpu().numpy()
        mval = (
            mval * (metrics[metric]["max"] - metrics[metric]["min"])
            + metrics[metric]["min"]
        )
        ret[metric] = mval

    ret["md"] = ret["mufd"] / ret["fof2"]
    return ret


def run_observation_diffusion(model, ts, essn, df_pred, num_samples=5):
    """Run observation-conditioned DiT inference.

    Uses cross-attention to sparse observations instead of gradient-based fitting.
    """
    params = model_params[model]

    # Load the observation-conditioned model
    if "dm" not in params:
        params["dm"] = ObservationConditionedDiT.load_from_checkpoint(
            params["diffusion_checkpoint"]
        ).to(device="cuda")
        params["dm"].eval()
    dm = params["dm"]

    max_obs = dm.hparams.max_observations

    # Process station data into observations
    observations_list = []
    for _, station in df_pred.iterrows():
        lat = station["station.latitude"] + 90  # Convert to [0, 180]
        lon = station["station.longitude"] + 180  # Convert to [0, 360]
        cs = float(station["cs"])  # Confidence score

        if not (0 <= lat < 181 and 0 <= lon < 361):
            continue

        # Build observation entry: [lat_norm, lon_norm, fof2, mufd, hmf2, weight_fof2, weight_mufd, weight_hmf2]
        obs = [0.0] * 8

        # Normalize location to [-1, 1]
        obs[0] = (lat / 181.0) * 2 - 1
        obs[1] = (lon / 361.0) * 2 - 1

        # Extract channel values and weights
        has_any_channel = False
        for metric_name, metric_info in metrics.items():
            ch = metric_info["ch"]
            min_val = metric_info["min"]
            max_val = metric_info["max"]

            if station[metric_name] is not None:
                normalized_value = (station[metric_name] - min_val) / (max_val - min_val)
                obs[2 + ch] = normalized_value
                obs[5 + ch] = cs  # Weight = confidence score
                has_any_channel = True

        # Only include observations with at least one channel
        if has_any_channel:
            observations_list.append(obs)

    # If more than max_observations, keep highest confidence
    if len(observations_list) > max_obs:
        observations_list.sort(key=lambda x: x[8], reverse=True)  # Sort by weight (cs)
        observations_list = observations_list[:max_obs]

    # Convert to tensor and pad to max_observations
    num_obs = len(observations_list)
    observations = torch.zeros(num_samples, max_obs, 8, device=dm.device)
    for i, obs in enumerate(observations_list):
        observations[:, i, :] = torch.tensor(obs, device=dm.device)

    # Setup time/SSN conditioning
    params_tensor = make_cfg_target(ts, essn).to(device=dm.device).expand(num_samples, -1)

    # Encode parameters and observations
    with torch.no_grad():
        param_embeds = dm.param_encoder.rope_enc(params_tensor)
        obs_embeds = dm.obs_encoder(observations)
        obs_weights = observations[..., 5:8]  # Extract per-channel weights for soft masking

        # Null conditioning for CFG
        null_param_embeds = torch.zeros_like(param_embeds)

        # Setup scheduler
        scheduler = diffusers.schedulers.DDPMScheduler(
            rescale_betas_zero_snr=False,
            prediction_type=dm.inference_scheduler.config.prediction_type,
        )
        scheduler.set_timesteps(params["steps"], device=dm.device)

        # Start from random noise
        x = torch.randn(num_samples, 4, 24, 48, device=dm.device)

        # Get observation guidance scale
        obs_guidance_scale = params.get("obs_guidance_scale", 1.0)

        # Denoising loop with CFG
        for t in scheduler.timesteps:
            torch.compiler.cudagraph_mark_step_begin()
            model_input = scheduler.scale_model_input(x, t)

            # Guided prediction (with time/SSN parameters and observations)
            noise_pred_guided = dm.model_forward_with_obs_guidance(
                model_input, t, param_embeds, obs_embeds, obs_weights,
                obs_guidance_scale=obs_guidance_scale
            )

            torch.compiler.cudagraph_mark_step_begin()
            # Unguided prediction (no parameters, with/without observations based on obs_guidance_scale)
            noise_pred_unguided = dm.model_forward_with_obs_guidance(
                model_input, t, null_param_embeds, obs_embeds, obs_weights,
                obs_guidance_scale=obs_guidance_scale
            )

            # Apply classifier-free guidance for parameters
            noise_pred = noise_pred_unguided + params["guidance_scale"] * (
                noise_pred_guided - noise_pred_unguided
            )

            # Denoise step
            x = scheduler.step(noise_pred, t, x).prev_sample

        # Decode final latents
        latents_crop = x[:, :, :23, :46]
        outs = dm.vae.decode(dm.scale_latents(latents_crop)).sample
        outs = scale_from_diffusion(outs)
        outs = outs[:, :, :181, :361]  # Crop to valid region
        outs = outs.clamp(0.0, 1.0)

        # Force 180E and 180W to be equal
        outs[..., 360] = outs[..., 0]

    ensemble = torch.quantile(outs, 0.5, dim=0)

    ret = {}
    for metric in metrics:
        mval = ensemble[metrics[metric]["ch"], ...].detach().cpu().numpy()
        mval = (
            mval * (metrics[metric]["max"] - metrics[metric]["min"])
            + metrics[metric]["min"]
        )
        ret[metric] = mval

    ret["md"] = ret["mufd"] / ret["fof2"]
    return ret


def assimilate(run_id, ts, holdout, model):
    df_cur = get_current()
    df_pred = get_pred(run_id, ts)
    irimap = get_irimap(run_id, ts)

    if holdout:
        holdouts = get_holdouts(run_id)
        df_pred = filter_holdouts(df_pred, holdouts)

    bio = io.BytesIO()
    h5 = h5py.File(bio, "w")

    h5.create_dataset("/essn/ssn", data=irimap["/essn/ssn"])
    h5.create_dataset("/essn/sfi", data=irimap["/essn/sfi"])
    h5.create_dataset("/ts", data=irimap["/ts"])
    h5.create_dataset("/stationdata/curr", data=df_cur.to_json(orient="records"))
    h5.create_dataset("/stationdata/pred", data=df_pred.to_json(orient="records"))
    h5.create_dataset(
        "/maps/foe", data=irimap["/maps/foe"], **hdf5plugin.SZ(absolute=0.001)
    )
    h5.create_dataset(
        "/maps/gyf", data=irimap["/maps/gyf"], **hdf5plugin.SZ(absolute=0.001)
    )

    # Choose which diffusion method to use based on model mode
    params = model_params[model]
    if params.get("mode") == "repaint":
        diffusion_out = run_repaint(model, ts, irimap["/essn/ssn"][()], df_pred)
    elif params.get("mode") == "observation_conditioned":
        num_samples = params.get("num_samples", 5)
        diffusion_out = run_observation_diffusion(model, ts, irimap["/essn/ssn"][()], df_pred, num_samples=num_samples)
    else:
        diffusion_out = run_diffusion(model, ts, irimap["/essn/ssn"][()], df_pred)

    for metric in ("fof2", "hmf2", "mufd", "md"):
        h5.create_dataset(
            f"/maps/{metric}",
            data=diffusion_out[metric],
            **hdf5plugin.SZ(absolute=0.001),
        )

    h5.close()
    return bio.getvalue()


app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate():
    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (
        os.getenv("DB_NAME"),
        os.getenv("DB_USER"),
        os.getenv("DB_HOST"),
        os.getenv("DB_PASSWORD"),
    )
    con = psycopg.connect(dsn)

    run_id = int(request.form.get("run_id", -1))
    tgt = int(request.form.get("target", None))
    holdout = bool(request.form.get("holdout", False))
    model = request.form.get("model", "latent_cfg_vpredict")

    tm = datetime.datetime.fromtimestamp(float(tgt), tz=datetime.timezone.utc)
    dataset = assimilate(run_id, tgt, holdout, model)

    with con.cursor() as cur:
        cur.execute(
            """insert into assimilated (time, run_id, dataset)
                    values (%s, %s, %s)
                    on conflict (run_id, time) do update set dataset=excluded.dataset""",
            (tm, run_id, dataset),
        )

        con.commit()
    con.close()

    return make_response("OK\n")
