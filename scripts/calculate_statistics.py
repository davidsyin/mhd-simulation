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
    Assumes the dataset is not split.
    Calculate statistics for normalization using only the training split.

    Args:
        data_path: Path to the dataset file
        config_path: Path to the config file containing training parameters
        output_file: Path to save the calculated statistics (.npz format)

    Returns:
        Dictionary with calculated statistics
    """
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)

    data_path = (
        config["dataset_params"]["data_dir"] + config["dataset_params"]["data_file"]
    )
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)

    # Assuming shape (num_samples, timesteps, nx, ny, channels)
    num_samples, timesteps, nx, ny, channels = data.shape
    print(f"Dataset shape: {data.shape}")

    # Get training ratio from config
    dataset_params = config.get("dataset_params", {})
    num_train = dataset_params.get("num_train", 0)
    num_test = dataset_params.get("num_test", 0)

    if num_train <= 0 or num_test <= 0:
        raise ValueError("Invalid 'num_train' or 'num_test' values in config file")

    train_ratio = num_train / (num_train + num_test)
    print(f"Using train_ratio: {train_ratio}")

    # Use the same random seed as in your dataset.py
    seed = dataset_params.get("seed", 42)  # Default seed if not specified
    print(f"Using random seed: {seed}")

    # Apply the same permutation logic as in dataset.py
    np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    train_size = int(train_ratio * num_samples)
    train_indices = indices[:train_size]

    # # Debug print
    # print(f"indices: {indices}")

    print(f"Using {len(train_indices)} samples for statistics calculation")

    # Select only training data
    train_data = data[train_indices]

    # Reshape data to put channels as first dim after samples
    # (num_train_samples, channels, timesteps, nx, ny)
    train_data_reshaped = np.transpose(train_data, (0, 4, 1, 2, 3))

    # Flatten all dimensions except for sample and channel
    train_data_flat = train_data_reshaped.reshape(len(train_indices), channels, -1)

    # Calculate statistics on training data only
    mean = np.mean(train_data_flat, axis=(0, 2))
    std = np.std(train_data_flat, axis=(0, 2))
    min_val = np.min(train_data_flat, axis=(0, 2))
    max_val = np.max(train_data_flat, axis=(0, 2))

    # Print statistics summary
    print("Calculated statistics from training data only:")
    print(f"  Mean values: {mean}")
    print(f"  Std values: {std}")
    print(f"  Min values: {min_val}")
    print(f"  Max values: {max_val}")

    if config["normalization_params"]["stats_file"]:
        output_file = config["normalization_params"]["stats_file"]

    # Save statistics to file if specified
    if output_file:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Make sure the file has .npz extension
        if not output_file.endswith(".npz"):
            output_file = f"{os.path.splitext(output_file)[0]}.npz"

        # Save all statistics in a single .npz file
        np.savez(output_file, mean=mean, std=std, min_val=min_val, max_val=max_val)
        print(f"Statistics saved to {output_file}")

    # Return as dictionary for compatibility
    return {"mean": mean, "std": std, "min_val": min_val, "max_val": max_val}


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
        default="normalization_stats.npz",
        help="Path to save statistics (.npz file)",
    )

    args = parser.parse_args()
    calculate_training_statistics(args.config, args.output)
