"""
DataModule for loading forecast sequences from HDF5.

Handles:
1. Loading (t, t-1) map pairs from forecast period
2. Sampling observations from historical window with age
3. Map padding to match VAE input size
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning as L
import h5py
import hdf5plugin  # Required for SZ compression
import numpy as np
import math


class ForecastDataset(Dataset):
    """
    Dataset for loading forecast sequence pairs.

    Each sequence has 48 timesteps:
    - Hours 0-23: Historical period (24 hours before present)
    - Hour 24: Present time
    - Hours 25-47: Forecast period (23 forecast steps)

    For training, we sample (t, t-1) pairs from the forecast period.
    """

    def __init__(
        self,
        hdf5_path,
        num_observations=50,
        split='train',
        train_fraction=0.9,
        seed=42,
    ):
        """
        Args:
            hdf5_path: Path to HDF5 file
            num_observations: Number of observations to sample per example
            split: 'train' or 'val'
            train_fraction: Fraction of sequences for training
            seed: Random seed for train/val split
        """
        self.hdf5_path = hdf5_path
        self.num_observations = num_observations
        self.split = split

        # Open HDF5 file to get sequence list
        with h5py.File(hdf5_path, 'r') as f:
            all_seq_ids = sorted(list(f['sequences'].keys()))
            total_sequences = len(all_seq_ids)

        # Train/val split
        np.random.seed(seed)
        indices = np.random.permutation(total_sequences)
        train_size = int(total_sequences * train_fraction)

        if split == 'train':
            self.sequence_ids = [all_seq_ids[i] for i in indices[:train_size]]
        else:
            self.sequence_ids = [all_seq_ids[i] for i in indices[train_size:]]

        # Each sequence has 23 forecast steps (hours 25-47)
        self.forecast_steps = 23

        print(f"ForecastDataset ({split}): {len(self.sequence_ids)} sequences, "
              f"{len(self)} total examples")

    def __len__(self):
        """Each sequence contributes 23 forecast steps"""
        return len(self.sequence_ids) * self.forecast_steps

    def pad_map(self, image):
        """
        Pad map from (3, 181, 361) to (3, 184, 368) to match VAE input size.

        Padding strategy (matching IRIData):
        - Latitude: Pad 3 rows at bottom (replicate last row)
        - Longitude: Wrap around (periodic boundary)
        """
        # Take first 360 columns (drop last column to make it periodic)
        padded_image = image[..., :360].clone()

        # Pad bottom 3 rows (replicate last row)
        padded_image = F.pad(padded_image, (0, 0, 0, 3), mode='replicate')

        # Wrap around first 8 columns for longitude periodicity
        padded_image = torch.cat((padded_image, padded_image[..., :8]), dim=-1)

        return padded_image

    def sample_observations_with_age(self, maps_sequence, forecast_timestep):
        """
        Sample observations from historical window [0, 24] with age information.

        Args:
            maps_sequence: (48, 3, 181, 361) full sequence maps
            forecast_timestep: Target timestep in [25, 47]

        Returns:
            observations: (N, 9) tensor [lat, lon, fof2, mufd, hmf2, w_fof2, w_mufd, w_hmf2, age_hours]
        """
        observations = []

        for _ in range(self.num_observations):
            # Sample age uniformly in [forecast_timestep-24, forecast_timestep]
            # This gives observations from the 24-hour historical window
            min_age = forecast_timestep - 24
            max_age = forecast_timestep
            age_hours = np.random.uniform(min_age, max_age)

            # Map age to historical timestep
            obs_timestep = int(forecast_timestep - age_hours)
            obs_timestep = np.clip(obs_timestep, 0, 24)  # Clamp to historical window

            # Sample spatial location (cosine-weighted latitude)
            u = np.random.rand()
            lat_degrees = np.arcsin(2 * u - 1) * (180 / np.pi)  # [-90, 90]
            lat_pixel = int((lat_degrees + 90) / 180 * 181)
            lat_pixel = np.clip(lat_pixel, 0, 180)

            lon_pixel = np.random.randint(0, 361)

            # Get values from historical map
            fof2 = maps_sequence[obs_timestep, 0, lat_pixel, lon_pixel]
            mufd = maps_sequence[obs_timestep, 1, lat_pixel, lon_pixel]
            hmf2 = maps_sequence[obs_timestep, 2, lat_pixel, lon_pixel]

            # Normalize location to [-1, 1]
            lat_norm = (lat_pixel / 181.0) * 2 - 1
            lon_norm = (lon_pixel / 361.0) * 2 - 1

            # Normalize values (matching expected ranges from IRI-2020)
            # foF2: [1.5, 15] MHz → [0, 1]
            # MUFD: [5, 45] MHz → [0, 1]
            # hmF2: [150, 450] km → [0, 1]
            fof2_norm = (fof2 - 1.5) / (15.0 - 1.5)
            mufd_norm = (mufd - 5.0) / (45.0 - 5.0)
            hmf2_norm = (hmf2 - 150.0) / (450.0 - 150.0)

            # Clamp to [0, 1] in case of out-of-range values
            fof2_norm = np.clip(fof2_norm, 0.0, 1.0)
            mufd_norm = np.clip(mufd_norm, 0.0, 1.0)
            hmf2_norm = np.clip(hmf2_norm, 0.0, 1.0)

            # Sample confidence (Beta(2, 1) distribution)
            confidence = np.random.beta(2.0, 1.0)

            # Channel availability (60% all, 25% two, 15% one)
            channel_scenario = np.random.rand()
            if channel_scenario < 0.60:  # All channels
                w_fof2 = confidence
                w_mufd = confidence
                w_hmf2 = confidence
            elif channel_scenario < 0.85:  # Two channels
                drop_ch = np.random.randint(0, 3)
                w_fof2 = 0.0 if drop_ch == 0 else confidence
                w_mufd = 0.0 if drop_ch == 1 else confidence
                w_hmf2 = 0.0 if drop_ch == 2 else confidence
            else:  # One channel
                keep_ch = np.random.randint(0, 3)
                w_fof2 = confidence if keep_ch == 0 else 0.0
                w_mufd = confidence if keep_ch == 1 else 0.0
                w_hmf2 = confidence if keep_ch == 2 else 0.0

            # Zero out values for absent channels
            fof2_norm = fof2_norm if w_fof2 > 0 else 0.0
            mufd_norm = mufd_norm if w_mufd > 0 else 0.0
            hmf2_norm = hmf2_norm if w_hmf2 > 0 else 0.0

            obs = [
                lat_norm, lon_norm,
                fof2_norm, mufd_norm, hmf2_norm,
                w_fof2, w_mufd, w_hmf2,
                age_hours
            ]
            observations.append(obs)

        return torch.tensor(observations, dtype=torch.float32)

    def normalize_map(self, map_data):
        """
        Normalize map channels to [0, 1] for VAE.

        Args:
            map_data: (3, H, W) tensor with [fof2, mufd, hmf2]

        Returns:
            Normalized (3, H, W) tensor in [0, 1]
        """
        # Scaling parameters from app/interpolate.py
        # fof2: [1.5, 15.0] MHz
        # mufd: [5.0, 45.0] MHz
        # hmf2: [150.0, 450.0] km

        normalized = map_data.clone()
        normalized[0] = (map_data[0] - 1.5) / (15.0 - 1.5)   # foF2
        normalized[1] = (map_data[1] - 5.0) / (45.0 - 5.0)   # MUFD
        normalized[2] = (map_data[2] - 150.0) / (450.0 - 150.0)  # hmF2

        # Clamp to [0, 1] in case of out-of-range values
        normalized = torch.clamp(normalized, 0.0, 1.0)

        return normalized

    def __getitem__(self, idx):
        """
        Get a single (t, t-1) pair from forecast period.

        Returns:
            dict with keys:
                - image_t: (3, 184, 368) current map (padded, normalized to [0,1])
                - image_t_minus_1: (3, 184, 368) previous map (padded, normalized to [0,1])
                - params: (4,) global parameters [secular, toy, tod, ssn_norm]
                - observations: (N, 9) observations with age
        """
        # Map flat index to (sequence_id, forecast_step)
        seq_idx = idx // self.forecast_steps
        forecast_step = idx % self.forecast_steps  # [0, 22]
        timestep = 25 + forecast_step  # [25, 47]

        seq_id = self.sequence_ids[seq_idx]

        # Load from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            seq = f['sequences'][seq_id]

            # Load maps
            map_t = torch.from_numpy(seq['maps'][timestep])  # (3, 181, 361)
            map_t_minus_1 = torch.from_numpy(seq['maps'][timestep - 1])  # (3, 181, 361)

            # Load params
            params_t = torch.from_numpy(seq['params'][timestep])  # (4,)

            # Load full sequence for observation sampling
            maps_sequence = seq['maps'][:]  # (48, 3, 181, 361)

        # Sample observations with age
        observations = self.sample_observations_with_age(maps_sequence, timestep)

        # Normalize maps to [0, 1] for VAE
        map_t = self.normalize_map(map_t)
        map_t_minus_1 = self.normalize_map(map_t_minus_1)

        # Pad maps
        image_t = self.pad_map(map_t)
        image_t_minus_1 = self.pad_map(map_t_minus_1)

        return {
            'image_t': image_t,
            'image_t_minus_1': image_t_minus_1,
            'params': params_t,
            'observations': observations,
        }


class ForecastDataModule(L.LightningDataModule):
    """
    LightningDataModule for forecast sequences.
    """

    def __init__(
        self,
        hdf5_path='data/forecast_sequences.h5',
        train_batch=64,
        val_batch=64,
        num_workers=32,
        num_observations=50,
        train_fraction=0.9,
        seed=42,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.num_workers = num_workers
        self.num_observations = num_observations
        self.train_fraction = train_fraction
        self.seed = seed

    def setup(self, stage=None):
        """Create train and val datasets"""
        self.train_dataset = ForecastDataset(
            hdf5_path=self.hdf5_path,
            num_observations=self.num_observations,
            split='train',
            train_fraction=self.train_fraction,
            seed=self.seed,
        )

        self.val_dataset = ForecastDataset(
            hdf5_path=self.hdf5_path,
            num_observations=self.num_observations,
            split='val',
            train_fraction=self.train_fraction,
            seed=self.seed,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True,
        )
