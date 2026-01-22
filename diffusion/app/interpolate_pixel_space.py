"""
Pixel-space inpainting for interpolation.

Key difference from interpolate_inpaint.py:
- Replace known regions in PIXEL SPACE instead of latent space
- This preserves the better parameter correlation (0.66 vs 0.42)
- Should improve spatial interpolation quality

The trade-off: More encode/decode operations (slower)
"""
import math
import torch
import torchvision as tv
import diffusers
import jsonargparse
from pathlib import Path
from models import ConditionedDiffusionModel, DiTDiffusionModel
from tqdm import tqdm
import torch.nn.functional as F
from util import scale_from_diffusion, scale_to_diffusion
import urllib.request
import json
import datetime
import h5py
import hdf5plugin
import io


def lat_lon_distance(lat1, lon1, lat2, lon2):
    lat1_rad = torch.deg2rad(torch.as_tensor(lat1) - 90.0)
    lon1_rad = torch.deg2rad(torch.as_tensor(lon1) - 180.0)
    lat2_rad = torch.deg2rad(torch.as_tensor(lat2) - 90.0)
    lon2_rad = torch.deg2rad(torch.as_tensor(lon2) - 180.0)

    cos_angle = torch.sin(lat1_rad) * torch.sin(lat2_rad) + torch.cos(
        lat1_rad
    ) * torch.cos(lat2_rad) * torch.cos(lon2_rad - lon1_rad)

    cos_angle = cos_angle.clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_angle))


lats, lons = torch.meshgrid(
    torch.arange(0, 184, dtype=torch.float32),
    torch.arange(0, 368, dtype=torch.float32),
    indexing="ij",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lats = lats.to(device)
lons = lons.to(device)


def get_h5():
    with urllib.request.urlopen("https://prop.kc2g.com/api/assimilated.h5") as res:
        content = res.read()
        bio = io.BytesIO(content)
        h5 = h5py.File(bio, "r")
        return h5


def dilations(lat, lon, dilate_by):
    dilated_masks = []
    distance = lat_lon_distance(lats, lons, lat, lon)
    for i in range(dilate_by + 1):
        mask = (i + 0.5 - distance).clip(0.0, 1.0)
        dilated_masks.append(mask)
    return dilated_masks


metrics = {
    "fof2": {"min": 1.5, "max": 15.0, "ch": 0},
    "mufd": {"min": 5.0, "max": 45.0, "ch": 1},
    "hmf2": {"min": 150.0, "max": 450.0, "ch": 2},
}


def main(
    diffuser_checkpoint: Path = Path("diffuser_checkpoint.ckpt"),
    model_type: str = "unet",
    num_timesteps: int = 20,
    num_samples: int = 9,
    seed: int = 0,
    guidance_scale: float = 5.0,
    max_dilate: int = 45,
    debug_images: bool = False,
    mask_strength: float = 1.0,
):
    """Pixel-space inpainting for spatial interpolation.

    Args:
        model_type: Model architecture - "unet" or "dit"
    """

    torch.manual_seed(seed)

    # Load appropriate model class
    if model_type == "dit":
        dm = DiTDiffusionModel.load_from_checkpoint(diffuser_checkpoint).to(device="cuda")
    elif model_type == "unet":
        dm = ConditionedDiffusionModel.load_from_checkpoint(diffuser_checkpoint).to(device="cuda")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'unet' or 'dit'")

    dm.eval()

    scheduler = diffusers.schedulers.DDPMScheduler(
        rescale_betas_zero_snr=True,
        prediction_type=dm.inference_scheduler.config.prediction_type,
    )
    scheduler.set_timesteps(num_timesteps, device=dm.device)
    print(f"Prediction type: {scheduler.config.prediction_type}")
    print(f"Using PIXEL SPACE inpainting (not latent space)")

    # Load station observations
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

    # Create observation targets and masks
    out_targets = [
        torch.zeros((3, 184, 368), device=dm.device) for _ in range(max_dilate + 1)
    ]
    dilated_masks = [
        torch.zeros((3, 184, 368), device=dm.device) for _ in range(max_dilate + 1)
    ]
    unweighted_masks = [
        torch.zeros((3, 184, 368), device=dm.device) for _ in range(max_dilate + 1)
    ]

    for station in station_data:
        lat = station["station.latitude"] + 90
        lon = station["station.longitude"] + 180
        cs = float(station["cs"])

        if not (0 <= lat < 181 and 0 <= lon < 361):
            continue

        pos_masks = dilations(lat, lon, max_dilate)

        for metric in metrics:
            if station[metric] is None:
                continue
            ch = metrics[metric]["ch"]
            min_val = metrics[metric]["min"]
            max_val = metrics[metric]["max"]
            normalized_value = (station[metric] - min_val) / (max_val - min_val)
            for i, m in enumerate(pos_masks):
                out_targets[i][ch, ...] += m * cs * normalized_value
                dilated_masks[i][ch, ...] += m * cs
                unweighted_masks[i][ch, ...] += m

    for i in range(max_dilate + 1):
        out_targets[i] /= torch.clip(dilated_masks[i], 1e-6, None)
        out_targets[i] = torch.clip(out_targets[i], 0.0, 1.0)
        dilated_masks[i] /= torch.clip(unweighted_masks[i], 1e-6, None)
        dilated_masks[i] = torch.clip(dilated_masks[i], 0.0, 1.0)
        if debug_images:
            tv.utils.save_image(out_targets[i].flip((1,)), f"out/target_{i:02d}.png")
            tv.utils.save_image(
                dilated_masks[i].flip((1,)), f"out/mask_dilated_{i:02d}.png"
            )

    # Crop to valid region (181x361)
    out_targets = [t[..., :181, :361] for t in out_targets]
    dilated_masks = [m[..., :181, :361] for m in dilated_masks]

    # Expand for batch
    out_targets = [t.unsqueeze(0).expand(num_samples, -1, -1, -1) for t in out_targets]
    dilated_masks = [m.unsqueeze(0).expand(num_samples, -1, -1, -1) for m in dilated_masks]

    # Setup conditioning
    dt = datetime.datetime.fromtimestamp(ts)
    year = dt.year
    toy = (
        dt - dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    ) / datetime.timedelta(days=365)
    tod = (
        dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)
    ) / datetime.timedelta(hours=24)

    guidance_target = torch.tensor(
        [
            (year - 2000.0 + toy) / 50.0,
            toy,
            tod,
            ssn / 100.0 - 1.0,
        ]
    )
    guidance_target = guidance_target.to(dm.device)
    guidance_target = guidance_target.expand(num_samples, -1)

    # DiT uses rope_enc (no scrambler), UNet uses full param_encoder (with scrambler)
    if model_type == "dit":
        encoded_target = dm.param_encoder.rope_enc(guidance_target)
    else:
        encoded_target = dm.param_encoder(guidance_target)

    null_target = torch.zeros_like(encoded_target)

    # Start from random noise in LATENT space
    x_latent = torch.randn((num_samples, 4, 24, 48), device=dm.device)

    # Denoising loop
    pbar = tqdm(enumerate(scheduler.timesteps), total=num_timesteps, desc="Pixel inpainting")

    for i, t in pbar:
        # Determine which mask to use
        progress = 1.0 - (i / num_timesteps)
        dilate_mask_by = int(round(progress * max_dilate))
        dilate_mask_by = min(max(dilate_mask_by, 0), max_dilate)

        mask_pixel = dilated_masks[dilate_mask_by]
        known_pixel = out_targets[dilate_mask_by]

        with torch.no_grad():
            # Denoise in latent space
            torch.compiler.cudagraph_mark_step_begin()
            model_input = scheduler.scale_model_input(x_latent, t)
            noise_pred_guided = dm.model(
                model_input, t, class_labels=encoded_target
            ).sample.clone()  # Clone to prevent CUDAGraphs overwrite

            torch.compiler.cudagraph_mark_step_begin()
            noise_pred_unguided = dm.model(
                model_input, t, class_labels=null_target
            ).sample.clone()  # Clone to prevent CUDAGraphs overwrite

            noise_pred = noise_pred_unguided + guidance_scale * (
                noise_pred_guided - noise_pred_unguided
            )

            # Get denoised latent
            x_latent_denoised = scheduler.step(noise_pred, t, x_latent).prev_sample

            # Decode to pixel space for masking
            # Crop latent to valid region (23x46) before decoding
            x_latent_crop = x_latent_denoised[:, :, :23, :46]
            x_pixel = dm.vae.decode(dm.scale_latents(x_latent_crop)).sample
            x_pixel = scale_from_diffusion(x_pixel)  # [0, 1]
            x_pixel = x_pixel[:, :, :181, :361]  # Crop to valid region

            effective_mask = mask_pixel * mask_strength

            # x_pixel is already denoised (predicted clean image)
            # So blend with clean observations directly
            x_pixel_combined = effective_mask * known_pixel + (1 - effective_mask) * x_pixel

            # Re-encode to latent space
            x_pixel_scaled = scale_to_diffusion(x_pixel_combined)
            x_pixel_padded = F.pad(x_pixel_scaled, (0, 7, 0, 3))  # Pad to 184x368
            x_latent_new = dm.vae.encode(x_pixel_padded).latents
            x_latent_new = dm.unscale_latents(x_latent_new)
            x_latent_new = F.pad(x_latent_new, (0, 2, 0, 1))  # Pad to 24x48

            x_latent = x_latent_new

        if debug_images and i % 5 == 0:
            # Crop to valid region for consistency with other scripts
            x_debug = x_pixel_combined[0, ..., :181, :361]
            tv.utils.save_image(
                x_debug.flip((1,)), f"out/step_{i:03d}.png"
            )

        pbar.set_postfix({"dilate": dilate_mask_by})

    # Final decode
    x_latent_crop = x_latent[:, :, :23, :46]
    outs = dm.vae.decode(dm.scale_latents(x_latent_crop)).sample
    outs = scale_from_diffusion(outs)
    outs = outs[..., :181, :361]

    image_grid = tv.utils.make_grid(outs, nrow=math.ceil(math.sqrt(num_samples)))

    filename = "out/generated_pixel_inpaint.png"
    tv.utils.save_image(image_grid.flip((1,)), filename)
    print(f"Generated images saved to {filename}")

    ensemble = torch.quantile(outs, 0.5, dim=0)
    tv.utils.save_image(ensemble.flip((1,)), "out/ensemble_pixel_inpaint.png")


if __name__ == "__main__":
    jsonargparse.CLI(main)
