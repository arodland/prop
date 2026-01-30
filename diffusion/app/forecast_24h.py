#!/usr/bin/env python3
"""
24-hour autoregressive ionosphere forecasting.

Given:
- Initial map at t=0 (present)
- Observations from past 23 hours (with ages)
- Target parameters for next 24 hours

Produces:
- 24 hourly maps for t+1 to t+24
"""

import torch
import numpy as np
import h5py
import hdf5plugin  # Required for SZ compression
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from models_forecast import ForecastObservationConditionedDiT


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = ForecastObservationConditionedDiT.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model.eval()
    model.to(device)
    return model


def normalize_map(map_data):
    """
    Normalize map channels to [0, 1] for VAE.

    Args:
        map_data: (3, H, W) tensor with [fof2, mufd, hmf2] in raw units

    Returns:
        Normalized (3, H, W) tensor in [0, 1]
    """
    normalized = map_data.clone()
    normalized[0] = (map_data[0] - 1.5) / (15.0 - 1.5)   # foF2
    normalized[1] = (map_data[1] - 5.0) / (45.0 - 5.0)   # MUFD
    normalized[2] = (map_data[2] - 150.0) / (450.0 - 150.0)  # hmF2
    normalized = torch.clamp(normalized, 0.0, 1.0)
    return normalized


def prepare_observations_for_forecast(observations_t0, hour_offset):
    """
    Age observations for forecasting at t+hour_offset.

    Args:
        observations_t0: (N, 9) observations at t=0 (with initial ages from past 23h)
        hour_offset: How many hours into the future we're forecasting (1-24)

    Returns:
        (N, 9) observations with updated ages
    """
    observations = observations_t0.clone()
    # Increment age by hour_offset
    observations[:, 8] += hour_offset
    return observations


def forecast_24h(
    model,
    initial_map,
    observations_t0,
    params_sequence,
    cfg_scale_obs=1.5,
    cfg_scale_prev=2.0,
    cfg_scale_joint=1.0,
    num_inference_steps=50,
    device='cuda',
):
    """
    Autoregressively forecast 24 hours.

    Args:
        model: ForecastObservationConditionedDiT model
        initial_map: (3, 184, 368) initial map at t=0
        observations_t0: (N, 9) observations from past 23h (already aged relative to t=0)
        params_sequence: (24, 4) target parameters for hours 1-24
        cfg_scale_obs: CFG scale for observations
        cfg_scale_prev: CFG scale for previous map
        cfg_scale_joint: CFG scale for joint conditioning
        num_inference_steps: Number of denoising steps
        device: Device to run on

    Returns:
        forecasts: List of 24 maps (each (3, 184, 368))
    """
    model.eval()

    forecasts = []
    current_map = initial_map.to(device)

    with torch.no_grad():
        for hour in range(24):
            print(f"Forecasting hour {hour + 1}/24...")

            # Get conditioning for this hour
            params_t = params_sequence[hour:hour+1].to(device)  # (1, 4)

            # Age observations for this forecast hour
            observations_t = prepare_observations_for_forecast(
                observations_t0, hour + 1
            ).unsqueeze(0).to(device)  # (1, N, 9)

            # Encode previous map to latents
            latents_prev = model.encode_images_to_latents(current_map.unsqueeze(0))  # (1, 4, 24, 48)

            # Encode parameters
            param_embeds = model.param_encoder.rope_enc(params_t)  # (1, 256)

            # Encode observations
            obs_embeds = model.obs_encoder(observations_t)  # (1, N, 768)
            obs_weights = observations_t[..., 5:8]  # (1, N, 3)

            # Generate with hierarchical CFG
            predicted_latents = model.generate_with_cfg(
                prev_map_latents=latents_prev,
                observations=observations_t,
                params=params_t,
                cfg_scale_obs=cfg_scale_obs,
                cfg_scale_prev=cfg_scale_prev,
                cfg_scale_joint=cfg_scale_joint,
                num_inference_steps=num_inference_steps,
            )  # (1, 4, 24, 48)

            # Decode to image space
            # Crop latents to (23, 46) before decoding
            predicted_latents_cropped = predicted_latents[..., :23, :46]
            predicted_map = model.decode_latents_to_images(predicted_latents_cropped)  # (1, 3, 184, 368)

            # Unpad map from (184, 368) to (181, 361)
            predicted_map = predicted_map[0, :, :181, :361]  # (3, 181, 361)

            forecasts.append(predicted_map.cpu())

            # Update current map for next iteration
            # Need to re-pad for next iteration
            predicted_map_padded = pad_map(predicted_map)  # (3, 184, 368)
            current_map = predicted_map_padded

    return forecasts


def pad_map(image):
    """
    Pad map from (3, 181, 361) to (3, 184, 368).

    Args:
        image: (3, 181, 361) tensor

    Returns:
        (3, 184, 368) padded tensor
    """
    import torch.nn.functional as F

    # Take first 360 columns
    padded = image[..., :360].clone()

    # Pad bottom 3 rows (replicate)
    padded = F.pad(padded, (0, 0, 0, 3), mode='replicate')

    # Wrap around first 8 columns
    padded = torch.cat((padded, padded[..., :8]), dim=-1)

    return padded


def visualize_forecast(
    forecasts,
    ground_truth=None,
    save_path=None,
    channel_names=['foF2', 'MUFD', 'hmF2'],
    horizons=[0, 5, 11, 23],  # Show hours 1, 6, 12, 24
):
    """
    Visualize forecast results.

    Args:
        forecasts: List of 24 maps (each (3, 181, 361))
        ground_truth: Optional list of 24 ground truth maps
        save_path: Optional path to save figure
        channel_names: Names of the 3 channels
        horizons: Which hours to visualize (0-indexed)
    """
    num_horizons = len(horizons)
    num_channels = 3

    if ground_truth is not None:
        fig, axes = plt.subplots(num_channels, num_horizons * 2, figsize=(4 * num_horizons * 2, 3 * num_channels))
    else:
        fig, axes = plt.subplots(num_channels, num_horizons, figsize=(4 * num_horizons, 3 * num_channels))

    for ch in range(num_channels):
        for i, h in enumerate(horizons):
            forecast_map = forecasts[h][ch].cpu().numpy()

            if ground_truth is not None:
                # Plot forecast
                ax = axes[ch, i * 2]
                im = ax.imshow(forecast_map, cmap='viridis', aspect='auto')
                ax.set_title(f'{channel_names[ch]} Forecast +{h+1}h')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                plt.colorbar(im, ax=ax)

                # Plot ground truth
                ax = axes[ch, i * 2 + 1]
                gt_map = ground_truth[h][ch].cpu().numpy()
                im = ax.imshow(gt_map, cmap='viridis', aspect='auto')
                ax.set_title(f'{channel_names[ch]} Truth +{h+1}h')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                plt.colorbar(im, ax=ax)
            else:
                # Plot forecast only
                ax = axes[ch, i] if num_channels > 1 else axes[i]
                im = ax.imshow(forecast_map, cmap='viridis', aspect='auto')
                ax.set_title(f'{channel_names[ch]} Forecast +{h+1}h')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='24-hour autoregressive ionosphere forecasting')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')

    # Input data
    parser.add_argument('--hdf5-path', type=str, required=True,
                        help='Path to HDF5 dataset with sequences')
    parser.add_argument('--seq-id', type=int, default=0,
                        help='Sequence ID to forecast (0-indexed)')

    # CFG parameters
    parser.add_argument('--cfg-scale-obs', type=float, default=1.5,
                        help='CFG scale for observations')
    parser.add_argument('--cfg-scale-prev', type=float, default=2.0,
                        help='CFG scale for previous map')
    parser.add_argument('--cfg-scale-joint', type=float, default=1.0,
                        help='CFG scale for joint conditioning')
    parser.add_argument('--num-steps', type=int, default=50,
                        help='Number of denoising steps')

    # Observations
    parser.add_argument('--num-observations', type=int, default=50,
                        help='Number of observations to sample')

    # Output
    parser.add_argument('--output-dir', type=str, default='forecasts',
                        help='Output directory for forecasts')
    parser.add_argument('--save-hdf5', action='store_true',
                        help='Save forecasts to HDF5')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)

    # Load sequence from HDF5
    print(f"Loading sequence {args.seq_id} from {args.hdf5_path}...")
    with h5py.File(args.hdf5_path, 'r') as f:
        seq_ids = sorted(list(f['sequences'].keys()))
        seq_id = seq_ids[args.seq_id]
        seq = f['sequences'][seq_id]

        # Hour 23 is present (t=0)
        initial_map = torch.from_numpy(seq['maps'][23]).float()  # (3, 181, 361)
        initial_map = normalize_map(initial_map)  # Normalize to [0, 1] for VAE
        initial_map = pad_map(initial_map)  # (3, 184, 368)

        # Ground truth for hours 24-47 (forecast targets)
        ground_truth = [
            normalize_map(torch.from_numpy(seq['maps'][24 + h]).float())
            for h in range(24)
        ]

        # Parameters for forecast hours
        params_sequence = torch.from_numpy(seq['params'][24:48]).float()  # (24, 4)

        # Sample observations from historical window (hours 0-22)
        # with ages relative to t=0 (hour 23)
        maps_sequence = seq['maps'][:]

    # Sample observations
    print(f"Sampling {args.num_observations} observations from historical window...")
    # Use the ForecastDataset logic for sampling
    from data.forecast_datamodule import ForecastDataset
    dataset = ForecastDataset(args.hdf5_path, num_observations=args.num_observations)
    observations_t0 = dataset.sample_observations_with_age(maps_sequence, forecast_timestep=23)

    # Run forecast
    print("Running 24-hour forecast...")
    forecasts = forecast_24h(
        model=model,
        initial_map=initial_map,
        observations_t0=observations_t0,
        params_sequence=params_sequence,
        cfg_scale_obs=args.cfg_scale_obs,
        cfg_scale_prev=args.cfg_scale_prev,
        cfg_scale_joint=args.cfg_scale_joint,
        num_inference_steps=args.num_steps,
        device=device,
    )

    print("Forecast complete!")

    # Save to HDF5
    if args.save_hdf5:
        output_path = output_dir / f'forecast_seq{args.seq_id:06d}.h5'
        print(f"Saving forecasts to {output_path}...")

        with h5py.File(output_path, 'w') as f:
            # Stack forecasts
            forecast_array = torch.stack(forecasts).numpy()  # (24, 3, 181, 361)
            f.create_dataset('forecasts', data=forecast_array, compression='gzip')

            # Save ground truth
            gt_array = torch.stack(ground_truth).numpy()  # (24, 3, 181, 361)
            f.create_dataset('ground_truth', data=gt_array, compression='gzip')

            # Save metadata
            f.attrs['seq_id'] = seq_id
            f.attrs['cfg_scale_obs'] = args.cfg_scale_obs
            f.attrs['cfg_scale_prev'] = args.cfg_scale_prev
            f.attrs['cfg_scale_joint'] = args.cfg_scale_joint
            f.attrs['num_steps'] = args.num_steps
            f.attrs['num_observations'] = args.num_observations

        print(f"Saved to {output_path}")

    # Visualize
    if args.visualize:
        vis_path = output_dir / f'forecast_seq{args.seq_id:06d}_vis.png'
        print(f"Creating visualization...")
        visualize_forecast(
            forecasts=forecasts,
            ground_truth=ground_truth,
            save_path=vis_path,
        )

    # Compute RMSE
    print("\nComputing RMSE by horizon...")
    rmse_by_horizon = []
    for h in range(24):
        forecast_map = forecasts[h]
        gt_map = ground_truth[h]
        rmse = torch.sqrt(torch.mean((forecast_map - gt_map) ** 2))
        rmse_by_horizon.append(rmse.item())

    # Print results
    print("\nRMSE by forecast horizon:")
    for h in [0, 5, 11, 23]:
        print(f"  +{h+1:2d}h: {rmse_by_horizon[h]:.4f}")

    print(f"\nMean RMSE: {np.mean(rmse_by_horizon):.4f}")


if __name__ == '__main__':
    main()
