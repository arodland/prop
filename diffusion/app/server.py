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
from models import DiffusionModel, ConditionedDiffusionModel, GuidanceModel
import torch.nn.functional as F
from util import scale_from_diffusion

model_params = {
    'latent_classifier': {
        'mode': 'classifier',
        'diffusion_checkpoint': '/checkpoints/diffusion/vdiffusion-averaged-v4.ckpt',
        'guidance_checkpoint': '/checkpoints/guidance/guidance--v_num=0-epoch=343-val_loss=7.9e-05.ckpt',
        'steps': 100,
        'guidance_scale': 15,
        'fit_scale': 250,
    },
    'latent_cfg_epsilon': {
        'mode': 'cfg',
        'diffusion_checkpoint': '/checkpoints/diffusion/cdiffusion-averaged-v7.ckpt',
        'steps': 100,
        'guidance_scale': 5,
        'fit_scale': 100,
    },
    'latent_cfg_vpredict': {
        'mode': 'cfg',
        'diffusion_checkpoint': '/checkpoints/diffusion/cdiffusion-averaged-v5.ckpt',
        'steps': 100,
        'guidance_scale': 5,
        'fit_scale': 100,
    },
}

metrics = {
    'fof2': { 'min': 1.5, 'max': 15.0, 'ch': 0 },
    'mufd': { 'min': 5.0, 'max': 45.0, 'ch': 1 },
    'hmf2': { 'min': 150.0, 'max': 450.0, 'ch': 2 },
}

geometry_scale = 45.
max_dilate = 44

def get_current():
    return jsonapi.get_data('http://localhost:%s/stations.json' % os.getenv('API_PORT'))

def get_pred(run_id, ts):
    return jsonapi.get_data('http://localhost:%s/pred.json?run_id=%d&ts=%d' % (os.getenv('API_PORT'), run_id, ts))

def get_holdouts(run_id):
    return json.get_data('http://localhost:%s/holdout?run_id=%d' % (os.getenv('API_PORT'), run_id))

def get_irimap(run_id, ts):
    return hdf5.get_data('http://localhost:%s/irimap.h5?run_id=%d&ts=%d' % (os.getenv('API_PORT'), run_id, ts))

def filter_holdouts(df, holdouts):
    if len(holdouts):
        holdout_station_ids = [ row['station']['id'] for row in holdouts ]
        for ii in holdout_station_ids:
            df = df.drop(df[df['station.id'] == ii].index)

    return df

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

def dilations(lats, lons, lat, lon, dilate_by):
    """Dilates the mask by a specified number of pixels."""
    dilated_masks = []

    distance = lat_lon_distance(lats, lons, lat, lon)

    for i in range(dilate_by + 1):
        mask = (i + 0.5 - distance).clip(0.0, 1.0)
        dilated_masks.append(mask)
    return dilated_masks

def create_targets(df_pred, num_samples, device):
    out_targets = [ torch.zeros((3, 184, 368), device=device) for _ in range(max_dilate + 1) ]
    dilated_masks = [ torch.zeros((3, 184, 368), device=device) for _ in range(max_dilate + 1) ]
    unweighted_masks = [ torch.zeros((3, 184, 368), device=device) for _ in range(max_dilate + 1) ]

    lats, lons = torch.meshgrid(
        torch.arange(0, 184, dtype=torch.float32),
        torch.arange(0, 368, dtype=torch.float32),
        indexing="ij"
    )
    lats = lats.to(device)
    lons = lons.to(device)

    # Draw the known data for each station into the masks, at the 45 different dilation scales
    for _, station in df_pred.iterrows():
        lat = station['station.latitude'] + 90
        lon = station['station.longitude'] + 180
        cs = float(station['cs'])

        pos_masks = dilations(lats, lons, lat, lon, max_dilate)

        for metric in metrics:
            if station[metric] is None:
                continue
            ch = metrics[metric]['ch']
            min_val, max_val = metrics[metric]['min'], metrics[metric]['max']
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
    toy = (dt - dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)) / datetime.timedelta(days=365)
    toy = min(max(toy, 0.0), 1.0)
    tod = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)) / datetime.timedelta(hours=24)

    return torch.tensor([
        (year - 2000. + toy) / 50.,
        math.sin(toy * 2 * math.pi),
        math.cos(toy * 2 * math.pi),
        math.sin(tod * 2 * math.pi),
        math.cos(tod * 2 * math.pi),
        essn / 100.0 - 1.0,
    ])

def run_diffusion(model, ts, essn, df_pred, num_samples=5):
    params = model_params[model]
    cfg = params['mode'] == 'cfg'

    if cfg:
        if 'dm' not in params:
            params['dm'] = ConditionedDiffusionModel.load_from_checkpoint(
                params['diffusion_checkpoint']).to(device='cuda')
            params['dm'].eval()
        dm = params['dm']
    else:
        if 'dm' not in params:
            params['dm'] = DiffusionModel.load_from_checkpoint(params['diffusion_checkpoint']).to(device='cuda')
            params['dm'].eval()
        dm = params['dm']
        if 'gm' not in params:
            params['gm'] = GuidanceModel.load_from_checkpoint(params['guidance_checkpoint']).to(device='cuda')
            params['gm'].eval()
        gm = params['gm']

    scheduler = diffusers.schedulers.DDPMScheduler(rescale_betas_zero_snr=True)
    scheduler.set_timesteps(params['steps'], device=dm.device)

    targets, masks = create_targets(df_pred, device=dm.device, num_samples=num_samples)

    guidance_target = make_guidance_target(ts, essn).to(device=dm.device)
    if cfg:
        encoded_target = dm.param_encoder(guidance_target).expand(num_samples, -1)
        null_target = torch.zeros_like(encoded_target)

    guidance_target = guidance_target.expand(num_samples, -1)

    x = torch.randn((num_samples, 4, 24, 48), device=dm.device)

    zero_dilate_timestep = params['steps'] if cfg else params['steps'] * 0.95

    for i, t in enumerate(scheduler.timesteps):
        torch.compiler.cudagraph_mark_step_begin()
        model_input = scheduler.scale_model_input(x, t)

        # Make a denoising step, with the appropriate sort of guidance
        if cfg:
            with torch.no_grad():
                noise_pred_guided = dm.model(model_input, t, class_labels=encoded_target).sample
                noise_pred_unguided = dm.model(model_input, t, class_labels=null_target).sample
                noise_pred = noise_pred_unguided + params['guidance_scale'] * (noise_pred_guided - noise_pred_unguided)
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
        wrap_loss_lon = torch.var(x0_decoded[..., 0, :]) + torch.var(x0_decoded[..., 180, :])
        loss = geometry_scale * (wrap_loss_lat + wrap_loss_lon)

        # Calculate guidance loss if doing classifier guidance
        if not cfg:
            guidance_out = gm.model(x0_decoded)
            guidance_loss = F.mse_loss(guidance_out, guidance_target)
            loss += guidance_loss * params['guidance_scale']

        # Fetch the fit target/mask for the current timestep
        dilate_mask_by = int(round(((zero_dilate_timestep - i) * max_dilate) / zero_dilate_timestep))
        dilate_mask_by = 0 if dilate_mask_by < 0 else dilate_mask_by
        mask_dilated = masks[dilate_mask_by]
        out_target = targets[dilate_mask_by]

        # And apply the target-fitting loss
        if cfg or i <= zero_dilate_timestep:
            fit_loss = (x0_decoded - out_target).pow(2).mul(mask_dilated).sum() / mask_dilated.sum()
            loss += fit_loss * params['fit_scale']

        print(i, loss)

        # Backward the loss and take a step to decrease it
        grad = -torch.autograd.grad(loss, x)[0]
        x = x.detach() + grad
        x = scheduler.step(noise_pred, t, x).prev_sample

    outs = scale_from_diffusion(dm.vae.decode(dm.scale_latents(x)).sample)
    outs = outs[..., :181, :361]
    # Force 180E and 180W to be equal
    outs[..., 360] = outs[..., 0]

    ensemble = torch.quantile(outs, 0.5, dim=0)

    ret = {}
    for metric in metrics:
        mval = ensemble[metrics[metric]['ch'], ...].detach().cpu().numpy()
        mval = mval * (metrics[metric]['max'] - metrics[metric]['min']) + metrics[metric]['min']
        ret[metric] = mval

    ret['md'] = ret['mufd'] / ret['fof2']
    return ret

def assimilate(run_id, ts, holdout, model):
    df_cur = get_current()
    df_pred = get_pred(run_id, ts)
    irimap = get_irimap(run_id, ts)

    if holdout:
        holdouts = get_holdouts(run_id)
        df_pred = filter_holdouts(df_pred, holdouts)

    bio = io.BytesIO()
    h5 = h5py.File(bio, 'w')

    h5.create_dataset('/essn/ssn', data=irimap['/essn/ssn'])
    h5.create_dataset('/essn/sfi', data=irimap['/essn/sfi'])
    h5.create_dataset('/ts', data=irimap['/ts'])
    h5.create_dataset('/stationdata/curr', data=df_cur.to_json(orient='records'))
    h5.create_dataset('/stationdata/pred', data=df_pred.to_json(orient='records'))
    h5.create_dataset('/maps/foe', data=irimap['/maps/foe'], **hdf5plugin.SZ(absolute=0.001))
    h5.create_dataset('/maps/gyf', data=irimap['/maps/gyf'], **hdf5plugin.SZ(absolute=0.001))

    diffusion_out = run_diffusion(model, ts, irimap['/essn/ssn'][()], df_pred)

    for metric in ('fof2', 'hmf2', 'mufd', 'md'):
        h5.create_dataset(f"/maps/{metric}", data=diffusion_out[metric], **hdf5plugin.SZ(absolute=0.001))

    h5.close()
    return bio.getvalue()

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    dsn = "dbname='%s' user='%s' host='%s' password='%s'" % (
        os.getenv("DB_NAME"), os.getenv("DB_USER"), os.getenv("DB_HOST"), os.getenv("DB_PASSWORD"))
    con = psycopg.connect(dsn)

    run_id = int(request.form.get('run_id', -1))
    tgt = int(request.form.get('target', None))
    holdout = bool(request.form.get('holdout', False))
    model = request.form.get('model', 'latent_cfg_vpredict')

    tm = datetime.datetime.fromtimestamp(float(tgt), tz=datetime.timezone.utc)
    dataset = assimilate(run_id, tgt, holdout, model)

    with con.cursor() as cur:
        cur.execute("""insert into assimilated (time, run_id, dataset)
                    values (%s, %s, %s)
                    on conflict (run_id, time) do update set dataset=excluded.dataset""",
                    (tm, run_id, dataset)
                    )

        con.commit()
    con.close()

    return make_response("OK\n")
