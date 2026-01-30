#!/usr/bin/env python3
"""
Generate forecast dataset for 24-hour autoregressive ionosphere prediction.

Creates HDF5 dataset with 48-hour sequences (24h history + 24h forecast)
using IRI-2020 model with realistic SSN temporal evolution.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool, cpu_count
import argparse

import numpy as np
import h5py
import hdf5plugin
from tqdm import tqdm


def generate_single_map(ssn, tm, irimap_binary='/build/irimap'):
    """
    Generate a single IRI-2020 map for given SSN and time.

    Args:
        ssn: Sunspot number (float)
        tm: datetime object (UTC)
        irimap_binary: Path to irimap executable

    Returns:
        dict with keys 'fof2', 'mufd', 'hmf2', 'foe' (all 181x361 numpy arrays)
    """
    cmd = [irimap_binary, str(tm.year), str(tm.month), str(tm.day),
           str(tm.hour), str(tm.minute), str(tm.second), str(ssn)]

    iri = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    data = {}
    for key in ['fof2', 'mufd', 'hmf2', 'foe']:
        data[key] = np.zeros((181, 361), dtype=np.float32)

    line_count = 0
    for line in iri.stdout:
        try:
            lat_, lon_, nmf2_, fof2_, md_, mufd_, hmf2_, foe_ = [float(x) for x in line.split()]
            lat = round(lat_ + 90)
            lon = round(lon_ + 180)

            data['fof2'][lat, lon] = fof2_
            data['mufd'][lat, lon] = mufd_
            data['hmf2'][lat, lon] = hmf2_
            data['foe'][lat, lon] = foe_
            line_count += 1
        except Exception as e:
            print(f"Error parsing irimap output line: {line[:100]}, error: {e}", file=sys.stderr, flush=True)
            continue

    # Get stderr output and return code
    _, stderr_output = iri.communicate()
    return_code = iri.returncode

    if line_count == 0:
        raise RuntimeError(
            f"irimap produced no output for time={tm}, ssn={ssn}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {return_code}\n"
            f"Stderr: {stderr_output[:500]}"
        )

    if line_count < 60000:  # Should be ~65000 lines for 181x361 grid
        print(
            f"Warning: irimap only produced {line_count} lines (expected ~65000) "
            f"for time={tm}, ssn={ssn}\n"
            f"Command: {' '.join(cmd)}",
            file=sys.stderr, flush=True
        )

    if return_code != 0:
        raise RuntimeError(
            f"irimap exited with code {return_code} for time={tm}, ssn={ssn}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr: {stderr_output[:500]}"
        )

    # Validate that the data contains reasonable values
    # foF2 should be mostly positive (1.5-15 MHz range)
    # Check if we have mostly invalid data (zeros or negatives)
    fof2_valid = np.sum(data['fof2'] > 0.5)  # Should have values > 0.5 MHz
    total_points = 181 * 361

    if fof2_valid < total_points * 0.5:  # Less than 50% valid data
        raise RuntimeError(
            f"irimap produced invalid data for time={tm}, ssn={ssn}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Only {fof2_valid}/{total_points} points have valid foF2 values (> 0.5 MHz)\n"
            f"foF2 range: [{data['fof2'].min():.2f}, {data['fof2'].max():.2f}]\n"
            f"mufd range: [{data['mufd'].min():.2f}, {data['mufd'].max():.2f}]\n"
            f"hmF2 range: [{data['hmf2'].min():.2f}, {data['hmf2'].max():.2f}]"
        )

    return data


def evolve_ssn_realistic(current_ssn, current_regime, dt=1.0, mu=100.0):
    """
    Evolve SSN using regime-switching OU with rare jumps.
    Based on analysis of real SSN data (6h series, 2000 days).

    Args:
        current_ssn: Current sunspot number
        current_regime: 'quiet' or 'active'
        dt: Time step in hours
        mu: Long-term mean (100)

    Returns:
        (new_ssn, new_regime) tuple

    Model:
        - Quiet regime (50% of time): sigma=1.0, theta=0.003
        - Active regime (50% of time): sigma=3.0, theta=0.002
        - Regime switching: 5% probability per hour
        - Rare jumps: 0.5% probability per hour, size~N(0,10^2)

    Expected statistics:
        - Hourly change std: ~2.3
        - Autocorr(24h): ~0.94
        - 95th percentile |change|: ~4.3
    """
    # Regime parameters
    if current_regime == 'quiet':
        theta = 0.003  # Faster mean reversion
        sigma = 1.0    # Low volatility
        switch_prob = 0.05
    else:  # active
        theta = 0.002  # Slower mean reversion
        sigma = 3.0    # High volatility
        switch_prob = 0.05

    # Regime switching (5% per hour)
    if np.random.rand() < switch_prob * dt:
        new_regime = 'active' if current_regime == 'quiet' else 'quiet'
    else:
        new_regime = current_regime

    # OU evolution
    dW = np.random.randn()
    drift = -theta * (current_ssn - mu) * dt
    diffusion = sigma * np.sqrt(dt) * dW
    new_ssn = current_ssn + drift + diffusion

    # Rare jumps (0.5% probability per hour)
    if np.random.rand() < 0.005 * dt:
        jump_size = np.random.randn() * 10.0
        new_ssn += jump_size

    # Clamp to reasonable bounds
    return np.clip(new_ssn, 0.0, 300.0), new_regime


def generate_sequence(seq_id, start_time, start_ssn, num_steps=48, irimap_binary='/build/irimap'):
    """
    Generate a single 48-hour forecast sequence.

    Sequence structure:
    - Hours 0-23: Historical period (t=-24 to t=-1)
    - Hour 24: Present time (t=0)
    - Hours 25-47: Forecast period (t=+1 to t=+23)

    Args:
        seq_id: Sequence identifier (for logging)
        start_time: datetime for hour 0 (24 hours before present)
        start_ssn: Initial SSN value
        num_steps: Total timesteps (default 48)
        irimap_binary: Path to irimap executable

    Returns:
        dict with keys:
            - maps: (48, 3, 181, 361) array [fof2, mufd, hmf2]
            - params: (48, 4) array [secular, toy, tod, ssn_norm]
            - timestamps: (48,) array of Unix timestamps
            - ssn: (48,) array of raw SSN values
    """
    maps = []
    params_list = []
    timestamps = []
    ssn_values = [start_ssn]

    # Initialize regime (randomly start in quiet or active)
    current_regime = 'quiet' if np.random.rand() < 0.75 else 'active'

    for hour in range(num_steps):
        current_time = start_time + timedelta(hours=hour)

        # Evolve SSN (except for first step)
        if hour > 0:
            current_ssn, current_regime = evolve_ssn_realistic(
                ssn_values[-1], current_regime, dt=1.0, mu=100.0
            )
            ssn_values.append(current_ssn)
        else:
            current_ssn = start_ssn

        # Generate IRI-2020 map
        try:
            map_data = generate_single_map(current_ssn, current_time, irimap_binary)
        except Exception as e:
            print(f"Error generating map for seq {seq_id}, hour {hour}: {e}", file=sys.stderr)
            raise

        # Stack channels: [fof2, mufd, hmf2]
        map_array = np.stack([
            map_data['fof2'],
            map_data['mufd'],
            map_data['hmf2']
        ], axis=0)  # (3, 181, 361)
        maps.append(map_array)

        # Compute normalized parameters (matching IRIData format)
        secular = (current_time.year - 2000) / 50.0  # Center at 2000, range ±50 years
        toy = current_time.timetuple().tm_yday / 365.0  # Time of year [0, 1]
        tod = (current_time.hour + current_time.minute / 60.0) / 24.0  # Time of day [0, 1]
        ssn_norm = current_ssn / 100.0 - 1.0  # Normalize to ~[-1, 2]

        params_list.append([secular, toy, tod, ssn_norm])
        timestamps.append(current_time.timestamp())

    return {
        'maps': np.array(maps, dtype=np.float32),  # (48, 3, 181, 361)
        'params': np.array(params_list, dtype=np.float32),  # (48, 4)
        'timestamps': np.array(timestamps, dtype=np.float64),
        'ssn': np.array(ssn_values, dtype=np.float32),  # (48,)
    }


def generate_sequence_wrapper(args):
    """Wrapper for multiprocessing."""
    seq_id, start_time, start_ssn, irimap_binary = args
    try:
        result = generate_sequence(seq_id, start_time, start_ssn, irimap_binary=irimap_binary)
        return seq_id, result
    except Exception as e:
        print(f"Failed to generate sequence {seq_id}: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return seq_id, None


def sample_sequence_parameters(num_sequences=10000, time_range=(2000, 2030), seed=42):
    """
    Sample parameters for generating diverse sequences.

    Strategy:
    - Sample uniformly from time range [2000, 2030]
    - Sample SSN from reasonable distribution [20, 200] with some high outliers
    - Ensure coverage of seasons (ToY) and time of day (ToD)

    Args:
        num_sequences: Number of sequences to generate
        time_range: (start_year, end_year) tuple
        seed: Random seed for reproducibility

    Returns:
        list of (start_time, start_ssn) tuples
    """
    np.random.seed(seed)

    start_year, end_year = time_range
    total_hours = (end_year - start_year) * 365 * 24

    params = []
    for i in range(num_sequences):
        # Sample random time in range
        # Ensure we have 48 hours of data (start at least 24h before end_year)
        max_offset_hours = total_hours - 48
        random_hours = np.random.randint(0, max_offset_hours)
        start_time = datetime(start_year, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(hours=int(random_hours))

        # Sample SSN: 70% from [30, 150], 20% low [0, 30], 10% high [150, 250]
        rand = np.random.rand()
        if rand < 0.70:
            start_ssn = np.random.uniform(30, 150)
        elif rand < 0.90:
            start_ssn = np.random.uniform(0, 30)
        else:
            start_ssn = np.random.uniform(150, 250)

        params.append((start_time, start_ssn))

    return params


def write_sequence_to_hdf5(h5file, seq_id, sequence_data):
    """
    Write a single sequence to HDF5 file.

    Args:
        h5file: h5py.File object
        seq_id: Sequence identifier (integer)
        sequence_data: Dict from generate_sequence()
    """
    grp = h5file.create_group(f'sequences/{seq_id:06d}')

    # Use compression for large arrays
    grp.create_dataset('maps', data=sequence_data['maps'],
                       compression='gzip', compression_opts=4)
    grp.create_dataset('params', data=sequence_data['params'],
                       compression='gzip', compression_opts=4)
    grp.create_dataset('timestamps', data=sequence_data['timestamps'])
    grp.create_dataset('ssn', data=sequence_data['ssn'])


def generate_dataset(
    output_path,
    num_sequences=10000,
    num_workers=32,
    irimap_binary='/build/irimap',
    time_range=(2000, 2028),
    seed=42,
):
    """
    Generate complete forecast dataset.

    Args:
        output_path: Path to output HDF5 file
        num_sequences: Number of sequences to generate
        num_workers: Number of parallel workers
        irimap_binary: Path to irimap executable
        time_range: (start_year, end_year) for sampling
        seed: Random seed
    """
    print(f"Generating {num_sequences} sequences with {num_workers} workers...")
    print(f"Time range: {time_range[0]}-{time_range[1]}")
    print(f"IRI binary: {irimap_binary}")

    # Sample sequence parameters
    print("Sampling sequence parameters...")
    sequence_params = sample_sequence_parameters(num_sequences, time_range, seed)

    # Prepare arguments for multiprocessing
    worker_args = [
        (i, start_time, start_ssn, irimap_binary)
        for i, (start_time, start_ssn) in enumerate(sequence_params)
    ]

    # Create HDF5 file with metadata
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as h5f:
        # Write metadata
        meta = h5f.create_group('metadata')
        meta.attrs['num_sequences'] = num_sequences
        meta.attrs['timesteps_per_sequence'] = 48
        meta.attrs['time_delta_hours'] = 1.0
        meta.attrs['generation_date'] = datetime.now(timezone.utc).isoformat()
        meta.attrs['time_range'] = time_range
        meta.attrs['seed'] = seed

        # Create sequences group
        h5f.create_group('sequences')

    # Generate sequences in parallel and write each one immediately
    print(f"Generating sequences (this will take ~13 hours with {num_workers} workers)...")

    successful_count = 0

    with h5py.File(output_path, 'a') as h5f:
        with Pool(num_workers) as pool:
            # Iterate over results as they complete and write immediately
            # Use chunksize=1 to get results as soon as each one completes
            for seq_id, sequence_data in tqdm(
                pool.imap_unordered(generate_sequence_wrapper, worker_args, chunksize=1),
                total=num_sequences, smoothing=0,
                desc="Generating and writing sequences"
            ):
                if sequence_data is not None:
                    try:
                        write_sequence_to_hdf5(h5f, seq_id, sequence_data)
                        h5f.flush()  # Force write to disk
                        successful_count += 1
                    except Exception as e:
                        print(f"Error writing sequence {seq_id}: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"Warning: Sequence {seq_id} failed generation, skipping.", file=sys.stderr)

    # Compute final file size
    file_size_gb = output_path.stat().st_size / (1024**3)
    print(f"\nDataset generation complete!")
    print(f"Output: {output_path}")
    print(f"File size: {file_size_gb:.2f} GB")
    print(f"Successfully generated: {successful_count} / {num_sequences} sequences")


def validate_dataset(hdf5_path):
    """
    Validate generated dataset.

    Checks:
    - File can be opened
    - Expected number of sequences
    - Shape correctness
    - SSN autocorrelation (should match OU process)
    """
    print(f"\nValidating dataset: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as h5f:
        # Check metadata
        meta = h5f['metadata']
        num_sequences = meta.attrs['num_sequences']
        print(f"Expected sequences: {num_sequences}")

        # Check sequences group
        sequences = h5f['sequences']
        actual_sequences = len(sequences.keys())
        print(f"Actual sequences: {actual_sequences}")

        # Sample random sequence
        seq_ids = list(sequences.keys())
        if seq_ids:
            random_seq_id = np.random.choice(seq_ids)
            seq = sequences[random_seq_id]

            maps = seq['maps'][:]
            params = seq['params'][:]
            ssn = seq['ssn'][:]
            timestamps = seq['timestamps'][:]

            print(f"\nSample sequence {random_seq_id}:")
            print(f"  Maps shape: {maps.shape} (expected: (48, 3, 181, 361))")
            print(f"  Params shape: {params.shape} (expected: (48, 4))")
            print(f"  SSN shape: {ssn.shape} (expected: (48,))")
            print(f"  SSN range: [{ssn.min():.1f}, {ssn.max():.1f}]")
            print(f"  SSN mean: {ssn.mean():.1f}")
            print(f"  SSN std: {ssn.std():.1f}")

            # Compute SSN autocorrelation
            ssn_centered = ssn - ssn.mean()
            autocorr = np.correlate(ssn_centered, ssn_centered, mode='full')[len(ssn) - 1:] / np.sum(ssn_centered**2)

            print(f"\nSSN autocorrelation:")
            print(f"  Lag 1: {autocorr[1]:.3f} (expected: ~0.95 for OU process)")
            print(f"  Lag 6: {autocorr[6]:.3f} (expected: ~0.74)")
            print(f"  Lag 12: {autocorr[12]:.3f} (expected: ~0.55)")
            print(f"  Lag 24: {autocorr[24]:.3f} (expected: ~0.30)")

            # Check map value ranges
            fof2 = maps[:, 0, :, :]
            mufd = maps[:, 1, :, :]
            hmf2 = maps[:, 2, :, :]

            print(f"\nMap value ranges:")
            print(f"  foF2: [{fof2.min():.2f}, {fof2.max():.2f}] MHz")
            print(f"  MUFD: [{mufd.min():.2f}, {mufd.max():.2f}] MHz")
            print(f"  hmF2: [{hmf2.min():.2f}, {hmf2.max():.2f}] km")

        print("\nValidation complete!")


def main():
    parser = argparse.ArgumentParser(description='Generate forecast dataset for ionosphere prediction')
    parser.add_argument('--output', type=str, default='data/forecast_sequences.h5',
                        help='Output HDF5 file path')
    parser.add_argument('--num-sequences', type=int, default=10000,
                        help='Number of sequences to generate')
    parser.add_argument('--num-workers', type=int, default=32,
                        help='Number of parallel workers')
    parser.add_argument('--irimap-binary', type=str, default='/build/irimap',
                        help='Path to irimap executable')
    parser.add_argument('--start-year', type=int, default=2000,
                        help='Start year for time range')
    parser.add_argument('--end-year', type=int, default=2028,
                        help='End year for time range')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--validate-only', type=str, default=None,
                        help='Only validate existing dataset (provide path)')

    args = parser.parse_args()

    if args.validate_only:
        validate_dataset(args.validate_only)
    else:
        generate_dataset(
            output_path=args.output,
            num_sequences=args.num_sequences,
            num_workers=args.num_workers,
            irimap_binary=args.irimap_binary,
            time_range=(args.start_year, args.end_year),
            seed=args.seed,
        )

        # Validate after generation
        validate_dataset(args.output)


if __name__ == '__main__':
    main()
