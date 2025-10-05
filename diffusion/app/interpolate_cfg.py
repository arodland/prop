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
    lat1_rad = torch.deg2rad(torch.as_tensor(lat1) - 90.0)  # Adjust latitude to be from -90 to 90
    lon1_rad = torch.deg2rad(torch.as_tensor(lon1) - 180.0)  # Adjust longitude to be from -180 to 180
    lat2_rad = torch.deg2rad(torch.as_tensor(lat2) - 90.0)  # Adjust latitude to be from -90 to 90
    lon2_rad = torch.deg2rad(torch.as_tensor(lon2) - 180.0)  # Adjust longitude to be from -180 to 180

    # Spherical law of cosines
    cos_angle = (torch.sin(lat1_rad) * torch.sin(lat2_rad) +
                 torch.cos(lat1_rad) * torch.cos(lat2_rad) *
                 torch.cos(lon2_rad - lon1_rad))

    cos_angle = cos_angle.clamp(-1.0, 1.0)  # Ensure the value is within the valid range for acos

    # Return angular distance in degrees
    return torch.rad2deg(torch.acos(cos_angle))

lats, lons = torch.meshgrid(
    torch.arange(0, 184, dtype=torch.float32),
    torch.arange(0, 368, dtype=torch.float32),
    indexing="ij"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lats = lats.to(device)
lons = lons.to(device)

def get_h5():
    with urllib.request.urlopen("https://prop.kc2g.com/api/assimilated.h5") as res:
        content = res.read()
        bio = io.BytesIO(content)
        h5 = h5py.File(bio, 'r')
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
    "fof2": { "min": 1.5, "max": 15.0, "ch": 0 },
    "mufd": { "min": 5.0, "max": 45.0, "ch": 1 },
    "hmf2": { "min": 150.0, "max": 450.0, "ch": 2 },
}

def main(
    diffuser_checkpoint: Path = Path("diffuser_checkpoint.ckpt"),
    guidance_checkpoint: Path = None,
    num_timesteps: int = 20,
    num_samples: int = 9,
    seed: int = 0,
    guidance_scale: float = 5.0,
    fit_scale: float = 15.0,
    geometry_scale: float = 1.0,
    max_dilate: int = 45,
    debug_images: bool = False,
):
    """Generates images from a trained diffusion model."""

    torch.manual_seed(seed)
    dm = ConditionedDiffusionModel.load_from_checkpoint(diffuser_checkpoint).to(device="cuda")
    dm.eval()

    if guidance_checkpoint is None:
        check_guidance = False
    else:
        check_guidance = True
        gm = GuidanceModel.load_from_checkpoint(guidance_checkpoint).to(device="cuda")
        gm.eval()

    scheduler = diffusers.schedulers.DDPMScheduler(
        rescale_betas_zero_snr=True, prediction_type=dm.inference_scheduler.config.prediction_type)
    scheduler.set_timesteps(num_timesteps, device=dm.device)
    print(f"Prediction type: {scheduler.config.prediction_type}")

    with urllib.request.urlopen("https://prop.kc2g.com/renders/current/mufd-normal-now_station.json") as response:
        station_data = json.loads(response.read().decode())

    with urllib.request.urlopen("https://prop.kc2g.com/api/latest_run.json") as response:
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

    out_targets = [ torch.zeros((3, 184, 368), device=dm.device) for _ in range(max_dilate + 1) ]
    dilated_masks = [ torch.zeros((3, 184, 368), device=dm.device) for _ in range(max_dilate + 1) ]
    unweighted_masks = [ torch.zeros((3, 184, 368), device=dm.device) for _ in range(max_dilate + 1) ]

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
            tv.utils.save_image(dilated_masks[i].flip((1,)), f"out/mask_dilated_{i:02d}.png")

    out_targets = [ t.expand(num_samples, -1, -1, -1) for t in out_targets ]
    dilated_masks = [m.expand(num_samples, -1, -1, -1) for m in dilated_masks]

    dt = datetime.datetime.fromtimestamp(ts)
    year = dt.year
    toy = (dt - dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)) / datetime.timedelta(days=365)
    tod = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)) / datetime.timedelta(hours=24)

    guidance_target = torch.tensor([
        (year - 2000. + toy) / 50.,
        toy,
        tod,
        ssn / 100.0 - 1.0,
    ])
    guidance_target = guidance_target.to(dm.device)
    guidance_target = guidance_target.expand(num_samples, -1)
    encoded_target = dm.param_encoder(guidance_target)
    null_target = torch.zeros_like(encoded_target)

    x = torch.randn((num_samples, 4, 24, 48), device=dm.device)
    # x = x * (1 - mask) + ((out_target * 2.) - 1) * mask

    bar = tqdm(enumerate(scheduler.timesteps), total=num_timesteps)
    for i, t in bar:
        stats = {}

        torch.compiler.cudagraph_mark_step_begin()
        model_input = scheduler.scale_model_input(x, t)
        with torch.no_grad:
            noise_pred_guided = dm.model(model_input, t, class_labels=encoded_target).sample
            noise_pred_unguided = dm.model(model_input, t, class_labels=null_target).sample
            noise_pred = noise_pred_unguided + guidance_scale * (noise_pred_guided - noise_pred_unguided)

        x = x.detach().requires_grad_()
        x0 = scheduler.step(noise_pred, t, x).pred_original_sample
        x0_decoded = scale_from_diffusion(dm.vae.decode(dm.scale_latents(x0)).sample)

        # Cut down to 181x361 valid data, then re-pad to 184x368
        x0_decoded = x0_decoded[..., :181, :361]
        x0_decoded = F.pad(x0_decoded, (0, 7, 0, 3))
        if debug_images:
            tv.utils.save_image(x0_decoded[0, ...].flip((1,)), f"out/step_{i:03d}.png")

        if check_guidance:
            guidance_out = gm.model(x0_decoded)
            guidance_loss = F.mse_loss(guidance_out, guidance_target)
            stats["guidance_loss"] = guidance_loss.item()

        dilate_mask_by = int(round(((num_timesteps - i) * max_dilate) / num_timesteps))
        dilate_mask_by = 0 if dilate_mask_by < 0 else dilate_mask_by
        mask_dilated = dilated_masks[dilate_mask_by]
        out_target = out_targets[dilate_mask_by]
        fit_loss = (x0_decoded - out_target).pow(2).mul(mask_dilated).sum() / mask_dilated.sum()
        stats["dilate"] = dilate_mask_by
        stats["fit_loss"] = fit_loss.item()

        loss = fit_loss * fit_scale

        wrap_loss_lat = F.mse_loss(
            x0_decoded[..., 0],
            x0_decoded[..., 360],
        )
        wrap_loss_lon = torch.var(x0_decoded[..., 0, :]) + torch.var(x0_decoded[..., 180, :])
        stats["wlat"] = wrap_loss_lat.item()
        stats["wlon"] = wrap_loss_lon.item()
        loss += geometry_scale * (wrap_loss_lat + wrap_loss_lon)

        stats["loss"] = loss.item()
        bar.set_postfix(stats)

        grad = -torch.autograd.grad(loss, x)[0]

        x = x.detach() + grad
        x = scheduler.step(noise_pred, t, x).prev_sample

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
