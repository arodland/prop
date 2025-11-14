import math
import torch
import torchvision as tv
import diffusers
import jsonargparse
from pathlib import Path
from models import ConditionedDiffusionModel, GuidanceModel
from tqdm import tqdm
import torch.nn.functional as F
import contextlib
from util import scale_from_diffusion
import urllib.request
import json
import datetime
import h5py
import hdf5plugin
import io


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
    """Dilates the mask by a specified number of pixels."""
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
    guidance_checkpoint: Path = None,
    num_timesteps: int = 20,
    num_samples: int = 9,
    seed: int = 0,
    guidance_scale: float = 5.0,
    max_dilate: int = 45,
    debug_images: bool = False,
    jump_length: int = 10,
    jump_n_sample: int = 10,
):
    """Generates images from a trained diffusion model."""

    torch.manual_seed(seed)
    dm = ConditionedDiffusionModel.load_from_checkpoint(diffuser_checkpoint).to(
        device="cuda"
    )
    dm.eval()

    if guidance_checkpoint is None:
        check_guidance = False
    else:
        check_guidance = True
        gm = GuidanceModel.load_from_checkpoint(guidance_checkpoint).to(device="cuda")
        gm.eval()

    scheduler = diffusers.schedulers.DDPMScheduler(
        rescale_betas_zero_snr=True,
        prediction_type=dm.inference_scheduler.config.prediction_type,
    )
    scheduler.set_timesteps(num_timesteps, device=dm.device)
    print(f"Prediction type: {scheduler.config.prediction_type}")
    print(f"Scheduler timesteps (first 5): {scheduler.timesteps[:5]}")
    print(f"Scheduler timesteps (last 5): {scheduler.timesteps[-5:]}")
    print(f"Total scheduler steps: {len(scheduler.timesteps)}")

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

    reference = torch.zeros((3, 181, 361), device=dm.device)

    assim_h5 = get_h5()
    for metric in metrics:
        ch = metrics[metric]["ch"]
        min_val = metrics[metric]["min"]
        max_val = metrics[metric]["max"]
        data = assim_h5[f"/maps/{metric}"][:]
        data = torch.tensor(data, device=dm.device)
        data = (data - min_val) / (max_val - min_val)
        reference[ch, ...] = data

    tv.utils.save_image(reference.flip((1,)), "out/reference.png")

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

    out_targets = [t.expand(num_samples, -1, -1, -1) for t in out_targets]
    dilated_masks = [m.expand(num_samples, -1, -1, -1) for m in dilated_masks]

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
    encoded_target = dm.param_encoder(guidance_target)
    null_target = torch.zeros_like(encoded_target)

    # Start from random noise
    x = torch.randn((num_samples, 4, 24, 48), device=dm.device)

    # Encode the known data (targets) to latent space for inpainting
    # We need to prepare latent versions of our targets
    known_latents = []
    for i in range(max_dilate + 1):
        # Encode target to latent space
        target_data = out_targets[i][
            0:1, ...
        ]  # Take first sample, shape: (1, 3, 184, 368)
        target_data = (target_data * 2.0) - 1.0  # Scale to [-1, 1] for VAE
        with torch.no_grad():
            # AutoencoderTiny returns latents directly, not a distribution
            # This will be shape (1, 4, 23, 46) because 184÷8=23, 368÷8=46
            target_latent = dm.vae.encode(target_data).latents
            target_latent = dm.unscale_latents(target_latent)
            # Pad to 24x48 to match diffusion model latent size
            target_latent = F.pad(
                target_latent,
                (0, 48 - target_latent.shape[-1], 0, 24 - target_latent.shape[-2]),
            )
        known_latents.append(target_latent.expand(num_samples, -1, -1, -1))

    # Downsample masks to latent resolution
    # The actual data is 181x361, which needs to be padded to 184x368 for the model
    # In latent space: 184÷8=23, 368÷8=46, so latent is 23x46
    # But the diffusion model uses 24x48, so we need to pad in latent space
    latent_masks = []
    for i in range(max_dilate + 1):
        # Take the full mask (already 184x368 from earlier code)
        mask_full = dilated_masks[i][0:1, :, :, :]  # Shape: (1, 3, 184, 368)
        # Downsample to latent space using average pooling (8x reduction)
        latent_mask = F.avg_pool2d(
            mask_full, kernel_size=8, stride=8
        )  # Shape: (1, 3, 23, 46)
        # Pad to 24x48 to match the latent dimensions
        latent_mask = F.pad(
            latent_mask, (0, 48 - latent_mask.shape[-1], 0, 24 - latent_mask.shape[-2])
        )  # (1, 3, 24, 48)
        # Use only the first channel and expand to all samples
        latent_masks.append(
            latent_mask[:, 0:1, :, :].expand(num_samples, 1, -1, -1)
        )  # (num_samples, 1, 24, 48)

    # RePaint loop with proper resampling/jumping
    # We'll iterate through timesteps, and at each one optionally jump backward and forward
    step_count = 0
    i = 0  # Current index in scheduler.timesteps

    # Track which timesteps should trigger jumps
    jump_timesteps = set(range(0, num_timesteps - jump_length, jump_length))
    jump_counts = {j: jump_n_sample - 1 for j in jump_timesteps}

    pbar = tqdm(total=num_timesteps * jump_n_sample, desc="RePaint")

    while i < num_timesteps:
        t = scheduler.timesteps[i]

        # Determine which mask to use based on progress
        # i goes from 0 (high noise) to num_timesteps-1 (low noise)
        # We want dilate to go from max_dilate (high noise) to 0 (low noise)
        progress = 1.0 - (i / num_timesteps)
        dilate_mask_by = int(round(progress * max_dilate))
        dilate_mask_by = min(max(dilate_mask_by, 0), max_dilate)

        mask_latent = latent_masks[dilate_mask_by]
        known_latent = known_latents[dilate_mask_by]

        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()

            # Line 5 of Algorithm 1: Sample x_known from the known region at timestep t
            # x_known_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
            alpha_bar = scheduler.alphas_cumprod[t.long()]
            if i < num_timesteps - 1:
                noise = torch.randn_like(known_latent)
                x_known = (
                    torch.sqrt(alpha_bar) * known_latent
                    + torch.sqrt(1 - alpha_bar) * noise
                )
            else:
                # At final timestep, no noise
                x_known = known_latent

            # Lines 6-7: Denoise the unknown region
            model_input = scheduler.scale_model_input(x, t)
            noise_pred_guided = dm.model(
                model_input, t, class_labels=encoded_target
            ).sample
            noise_pred_unguided = dm.model(
                model_input, t, class_labels=null_target
            ).sample
            noise_pred = noise_pred_unguided + guidance_scale * (
                noise_pred_guided - noise_pred_unguided
            )
            x_unknown = scheduler.step(noise_pred, t, x).prev_sample

            # Line 8: Combine known and unknown regions
            x = mask_latent * x_known + (1 - mask_latent) * x_unknown

        # Check if we should resample (jump back) at this timestep
        if i in jump_counts and jump_counts[i] > 0 and i < num_timesteps - 1:
            # Jump backward by jump_length steps
            jump_counts[i] -= 1
            i_jump_back = min(i + jump_length, num_timesteps - 1)

            # Add noise to jump forward in diffusion time
            with torch.no_grad():
                for _ in range(jump_length):
                    if i < num_timesteps - 1:
                        t_fwd = scheduler.timesteps[min(i + 1, num_timesteps - 1)]
                        beta = scheduler.betas[t_fwd.long()]
                        noise = torch.randn_like(x)
                        x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise
                        # i += 1
                        # if i >= num_timesteps - 1:
                        #     break
        else:
            # Move to next timestep
            i += 1

        step_count += 1
        if debug_images:
            with torch.no_grad():
                x0_pred = dm.vae.decode(dm.scale_latents(x)).sample
                x0_pred = scale_from_diffusion(x0_pred)
                x0_pred = x0_pred[..., :181, :361]
                tv.utils.save_image(
                    x0_pred[0, ...].flip((1,)), f"out/step_{step_count:04d}.png"
                )

        pbar.update(1)
        pbar.set_postfix(
            {
                "i": i,
                "t": t.item() if torch.is_tensor(t) else t,
                "dilate": dilate_mask_by,
            }
        )

    pbar.close()

    outs = scale_from_diffusion(dm.vae.decode(dm.scale_latents(x)).sample)
    outs = outs[..., :181, :361]

    image_grid = tv.utils.make_grid(outs, nrow=math.ceil(math.sqrt(num_samples)))

    filename = "out/generated.png"
    tv.utils.save_image(image_grid.flip((1,)), filename)
    print(f"Generated images saved to {filename}")

    ensemble = torch.quantile(outs, 0.5, dim=0)
    tv.utils.save_image(ensemble.flip((1,)), "out/ensemble.png")
    stdev = torch.std(outs, dim=0) / ensemble
    stdev = (stdev - stdev.min()) / (stdev.max() - stdev.min())
    tv.utils.save_image(stdev.flip((1,)), "out/stdev.png")


if __name__ == "__main__":
    jsonargparse.CLI(main)
