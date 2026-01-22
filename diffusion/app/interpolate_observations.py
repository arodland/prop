#!/usr/bin/env python3
"""
Observation-conditioned interpolation using cross-attention DiT.

Unlike interpolate_cfg.py which uses gradient-based fitting, this script
uses the observation-conditioned model that learns to interpolate via
cross-attention to sparse observations.
"""
import math
import torch
import torchvision as tv
import diffusers
import jsonargparse
from pathlib import Path
from models import ObservationConditionedDiT
from tqdm import tqdm
from util import scale_from_diffusion
import urllib.request
import json
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metrics = {
    "fof2": {"min": 1.5, "max": 15.0, "ch": 0},
    "mufd": {"min": 5.0, "max": 45.0, "ch": 1},
    "hmf2": {"min": 150.0, "max": 450.0, "ch": 2},
}


def main(
    checkpoint: Path = Path("obs_dit_checkpoint.ckpt"),
    num_timesteps: int = 50,
    num_samples: int = 9,
    seed: int = 0,
    guidance_scale: float = 5.0,
    obs_guidance_scale: float = 1.0,
):
    """
    Generate ionosphere maps using observation-conditioned DiT.

    Args:
        checkpoint: Path to observation-conditioned DiT checkpoint
        num_timesteps: Number of denoising steps
        num_samples: Number of samples to generate
        seed: Random seed
        guidance_scale: Classifier-free guidance scale for time/SSN parameters
        obs_guidance_scale: Observation guidance scale
            - 1.0: Normal (use observations as-is)
            - > 1.0: Stricter interpolation (matches observations more closely)
            - < 1.0: More global structure (smoother, less exact at observations)
            - 0.0: Ignore observations entirely
    """

    torch.manual_seed(seed)

    # Load observation-conditioned model
    print(f"Loading checkpoint: {checkpoint}")
    model = ObservationConditionedDiT.load_from_checkpoint(checkpoint).to(device)
    model.eval()

    # Fetch current station observations
    print("Fetching station observations...")
    with urllib.request.urlopen(
        "https://prop.kc2g.com/renders/current/mufd-normal-now_station.json"
    ) as response:
        station_data = json.loads(response.read().decode())

    with urllib.request.urlopen(
        "https://prop.kc2g.com/api/latest_run.json"
    ) as response:
        latest_run = json.loads(response.read().decode())

    ts = latest_run["maps"][0]["ts"]
    ssn = latest_run["essn"]

    # Process station data into observations
    observations_list = []
    for station in station_data:
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

    print(f"Found {len(observations_list)} valid observations")

    # If more than max_observations, keep highest confidence
    max_obs = model.hparams.max_observations
    if len(observations_list) > max_obs:
        print(f"Trimming to {max_obs} highest confidence observations")
        observations_list.sort(key=lambda x: x[8], reverse=True)  # Sort by weight (cs)
        observations_list = observations_list[:max_obs]

    # Convert to tensor and pad to max_observations
    num_obs = len(observations_list)
    observations = torch.zeros(num_samples, max_obs, 8, device=device)
    for i, obs in enumerate(observations_list):
        observations[:, i, :] = torch.tensor(obs, device=device)

    print(f"Using {num_obs} observations (max: {max_obs})")

    # Setup time/SSN conditioning
    dt = datetime.datetime.fromtimestamp(ts)
    year = dt.year
    toy = (
        dt - dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    ) / datetime.timedelta(days=365)
    tod = (
        dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)
    ) / datetime.timedelta(hours=24)

    params = torch.tensor(
        [
            (year - 2000.0 + toy) / 50.0,
            toy,
            tod,
            ssn / 100.0 - 1.0,
        ],
        device=device,
    )
    params = params.expand(num_samples, -1)

    print(f"Time: {dt}")
    print(f"SSN: {ssn:.1f}")

    # Encode parameters and observations
    with torch.no_grad():
        param_embeds = model.param_encoder.rope_enc(params)
        obs_embeds = model.obs_encoder(observations)
        obs_weights = observations[..., 5:8]  # Extract per-channel weights for soft masking

        # Null conditioning for CFG
        null_param_embeds = torch.zeros_like(param_embeds)

        # Setup scheduler
        model.inference_scheduler.set_timesteps(num_timesteps, device=device)
        print(f"Prediction type: {model.inference_scheduler.config.prediction_type}")
        print(f"Using {num_timesteps} denoising steps")
        print(f"Guidance scale: {guidance_scale}")

        # Start from random noise
        x = torch.randn(num_samples, 4, 24, 48, device=device)

        # Denoising loop
        pbar = tqdm(model.inference_scheduler.timesteps, desc="Generating")
        for t in pbar:
            torch.compiler.cudagraph_mark_step_begin()
            model_input = model.inference_scheduler.scale_model_input(x, t)

            # Guided prediction (with parameters and observations)
            noise_pred_guided = model.model_forward_with_obs_guidance(
                model_input, t, param_embeds, obs_embeds, obs_weights,
                obs_guidance_scale=obs_guidance_scale
            )

            torch.compiler.cudagraph_mark_step_begin()
            # Unguided prediction (no parameters, with/without observations based on obs_guidance_scale)
            noise_pred_unguided = model.model_forward_with_obs_guidance(
                model_input, t, null_param_embeds, obs_embeds, obs_weights,
                obs_guidance_scale=obs_guidance_scale
            )

            # Apply classifier-free guidance for parameters
            noise_pred = noise_pred_unguided + guidance_scale * (
                noise_pred_guided - noise_pred_unguided
            )

            # Denoise step
            x = model.inference_scheduler.step(noise_pred, t, x).prev_sample

        # Decode final latents
        latents_crop = x[:, :, :23, :46]
        images = model.vae.decode(model.scale_latents(latents_crop)).sample
        images = scale_from_diffusion(images)
        images = images[:, :, :181, :361]  # Crop to valid region
        images = images.clamp(0.0, 1.0)

    # Save outputs
    image_grid = tv.utils.make_grid(images, nrow=math.ceil(math.sqrt(num_samples)))
    filename = "out/generated_obs.png"
    tv.utils.save_image(image_grid.flip((1,)), filename)
    print(f"Generated images saved to {filename}")

    # Ensemble (median)
    ensemble = torch.quantile(images, 0.5, dim=0)
    tv.utils.save_image(ensemble.flip((1,)), "out/ensemble_obs.png")
    print("Ensemble saved to out/ensemble_obs.png")

    # Standard deviation map
    stdev = torch.std(images, dim=0) / (ensemble + 1e-6)
    stdev = (stdev - stdev.min()) / (stdev.max() - stdev.min() + 1e-6)
    tv.utils.save_image(stdev.flip((1,)), "out/stdev_obs.png")
    print("Standard deviation saved to out/stdev_obs.png")


if __name__ == "__main__":
    jsonargparse.CLI(main)
