#!/usr/bin/env python
"""Calculate dataset statistics for normalization using only training data."""

import numpy as np
import argparse
import os
import sys

# Add project root to sys.path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config


def calculate_training_statistics(config_path, output_file=None):
    """
    Calculate statistics for normalization using only the training split.
    Supports both default and 'diffusion' config types.

    Args:
        config_path: Path to the config file containing training parameters
        output_file: Path to save the calculated statistics (.npz format)

    Returns:
        Dictionary with calculated statistics
    """
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)

    config_type = config.get("config_type", "default")

    if config_type == "diffusion":
        return _calculate_diffusion_statistics(config, output_file)
    else:
        return _calculate_default_statistics(config, output_file)


def _calculate_channel_statistics(data_flat):
    """
    Calculate channel-wise statistics from flattened data.

    Args:
        data_flat: Data tensor in shape (samples, channels, -1)

    Returns:
        Dictionary with calculated statistics
    """
    mean = np.mean(data_flat, axis=(0, 2))
    std = np.std(data_flat, axis=(0, 2))
    min_val = np.min(data_flat, axis=(0, 2))
    max_val = np.max(data_flat, axis=(0, 2))

    return {"mean": mean, "std": std, "min_val": min_val, "max_val": max_val}


def _save_statistics(stats_file, stats):
    """
    Save statistics to a file.

    Args:
        stats_file: Path to save the statistics file
        stats: Dictionary with statistics to save
    """
    os.makedirs(os.path.dirname(os.path.abspath(stats_file)), exist_ok=True)

    if not stats_file.endswith(".npz"):
        stats_file = f"{os.path.splitext(stats_file)[0]}.npz"

    np.savez(
        stats_file,
        mean=stats["mean"],
        std=stats["std"],
        min_val=stats["min_val"],
        max_val=stats["max_val"],
    )
    print(f"Statistics saved to {stats_file}")


def _calculate_default_statistics(config, output_file=None):
    """Calculate statistics for default configuration type."""
    data_path = (
        config["dataset_params"]["data_dir"] + config["dataset_params"]["data_file"]
    )
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)

    # Assuming shape (num_samples, timesteps, nx, ny, channels)
    num_samples, timesteps, nx, ny, channels = data.shape
    print(f"Dataset shape: {data.shape}")

    dataset_params = config.get("dataset_params", {})
    num_train = dataset_params.get("num_train", 0)

    sub_t = dataset_params.get("sub_t", 1)  # Default to 1 (no subsampling)
    sub_x = dataset_params.get("sub_x", 1)  # Default to 1 (no subsampling)
    print(f"Using temporal subsampling factor: {sub_t}")
    print(f"Using spatial subsampling factor: {sub_x}")

    if num_train <= 0:
        raise ValueError("Invalid 'num_train' value in config file")

    seed = dataset_params.get("seed", 42)
    print(f"Using random seed: {seed}")

    # Apply the same permutation logic as in dataset.py
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    train_size = num_train
    train_indices = indices[:train_size]

    print(f"Using {len(train_indices)} samples for statistics calculation")

    train_data = data[train_indices]

    # Apply subsampling
    if sub_t > 1 or sub_x > 1:
        print(f"Applying subsampling: sub_t={sub_t}, sub_x={sub_x}")
        train_data = train_data[:, ::sub_t, ::sub_x, ::sub_x, :]
        print(f"Data shape after subsampling: {train_data.shape}")

    # Reshape data to put channels as first dim after samples
    # (num_train_samples, channels, timesteps, nx, ny)
    train_data_reshaped = np.transpose(train_data, (0, 4, 1, 2, 3))

    # Flatten all dimensions except for sample and channel
    train_data_flat = train_data_reshaped.reshape(len(train_indices), channels, -1)

    stats = _calculate_channel_statistics(train_data_flat)

    print("Calculated statistics from training data only:")
    print(f"  Mean values: {stats['mean']}")
    print(f"  Std values: {stats['std']}")
    print(f"  Min values: {stats['min_val']}")
    print(f"  Max values: {stats['max_val']}")

    if config["normalization_params"]["stats_file"]:
        output_file = config["normalization_params"]["stats_file"]

    if output_file:
        _save_statistics(output_file, stats)

    return stats


def _calculate_diffusion_statistics(config, output_file=None):
    """Calculate statistics for diffusion configuration type."""
    train_path = config["dataset_params"]["train_path"]
    print(f"Loading diffusion data from {train_path}...")

    data = np.load(train_path, allow_pickle=True)

    data_dict = data.item()
    diff_inputs = data_dict["diff_inputs"]
    diff_targets = data_dict["diff_targets"]

    print(f"Inputs shape: {diff_inputs.shape}")
    print(f"Targets shape: {diff_targets.shape}")

    # Calculate statistics for inputs
    # For diffusion data, the shape is (samples, channels, timesteps, nx, ny)
    num_samples_inputs = diff_inputs.shape[0]
    input_channels = diff_inputs.shape[1]  # Channels are already in second position

    # No need to transpose as channels are already in the second dimension
    # Just flatten all dimensions except for sample and channel
    inputs_flat = diff_inputs.reshape(num_samples_inputs, input_channels, -1)

    input_stats = _calculate_channel_statistics(inputs_flat)

    num_samples_targets = diff_targets.shape[0]
    target_channels = diff_targets.shape[1]  # Channels are already in second position

    targets_flat = diff_targets.reshape(num_samples_targets, target_channels, -1)

    target_stats = _calculate_channel_statistics(targets_flat)

    print("Calculated input statistics:")
    print(f"  Mean values: {input_stats['mean']}")
    print(f"  Std values: {input_stats['std']}")
    print(f"  Min values: {input_stats['min_val']}")
    print(f"  Max values: {input_stats['max_val']}")

    print("Calculated target statistics:")
    print(f"  Mean values: {target_stats['mean']}")
    print(f"  Std values: {target_stats['std']}")
    print(f"  Min values: {target_stats['min_val']}")
    print(f"  Max values: {target_stats['max_val']}")

    inputs_stats_file = config["normalization_params"]["inputs_stats_file"]
    targets_stats_file = config["normalization_params"]["targets_stats_file"]

    # Check if the stats files are the same
    if inputs_stats_file == targets_stats_file:
        print("WARNING: Input and target stats files have the same name in the config.")
        base_path = os.path.splitext(inputs_stats_file)[0]
        inputs_stats_file = f"{base_path}_inputs.npz"
        targets_stats_file = f"{base_path}_targets.npz"
        print(f"Using modified filenames: {inputs_stats_file} and {targets_stats_file}")

    _save_statistics(inputs_stats_file, input_stats)
    _save_statistics(targets_stats_file, target_stats)

    return {"inputs": input_stats, "targets": target_stats}


def load_statistics(stats_file):
    """
    Load statistics from a .npz file.

    Args:
        stats_file: Path to the .npz file containing the statistics

    Returns:
        Dictionary with loaded statistics
    """
    if not stats_file.endswith(".npz"):
        raise ValueError(f"Expected .npz file, got {stats_file}")

    stats = np.load(stats_file)
    return {key: stats[key] for key in stats.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate dataset statistics from training data only"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save statistics (.npz file) - only used for default config type",
    )

    args = parser.parse_args()
    calculate_training_statistics(args.config, args.output)
