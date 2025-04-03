"""Shared fixtures for testing the NODs package."""

import os
import sys
import pytest
import torch
import numpy as np
import yaml

# Add the parent directory to the path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def device():
    """Return the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mini_config_path():
    """Return the path to the test configuration file."""
    return os.path.join(os.path.dirname(__file__), "configs", "test_config_tfno3d.yaml")


@pytest.fixture
def mini_data_path():
    """Return the path to the mini test dataset."""
    return os.path.join(os.path.dirname(__file__), "data", "mini_mhd_data.npy")


@pytest.fixture
def mini_data(mini_data_path):
    """Return the loaded data from mini_data_path."""
    return np.load(mini_data_path)


@pytest.fixture
def generate_mini_mhd_data(mini_data_path):
    """
    Generate a small MHD dataset for testing.

    Creates a dataset with shape (10, 21, 32, 32, 3) representing:
    - 10 samples
    - 21 timesteps
    - 32x32 spatial grid
    - 3 channels (u_x, u_y, A)
    """
    # Check if the file already exists
    if os.path.exists(mini_data_path):
        return mini_data_path

    # Make sure directory exists
    os.makedirs(os.path.dirname(mini_data_path), exist_ok=True)

    # Generate small random dataset with proper shape and range
    # Shape: (num_samples, timesteps, nx, ny, channels)
    np.random.seed(42)  # For reproducibility

    # Number of samples, timesteps, grid size, and channels
    num_samples = 10
    timesteps = 21
    nx, ny = 32, 32
    channels = 3

    # Generate random data with physically plausible ranges
    # Channel 0: u_x velocity component (-1 to 1)
    # Channel 1: u_y velocity component (-1 to 1)
    # Channel 2: A magnetic potential (smaller scale ~1e-4)
    data = np.zeros((num_samples, timesteps, nx, ny, channels))

    # Generate velocity fields with some structure (not just noise)
    for sample in range(num_samples):
        # Create structured velocity field
        x = np.linspace(0, 2 * np.pi, nx)
        y = np.linspace(0, 2 * np.pi, ny)
        X, Y = np.meshgrid(x, y)

        # Base flow pattern
        u_x_base = 0.5 * np.sin(X) * np.cos(Y)
        u_y_base = -0.5 * np.cos(X) * np.sin(Y)

        # Create time evolution with slight randomness
        for t in range(timesteps):
            # Time factor to simulate evolution
            time_factor = 0.1 * t

            # Add time-dependent evolution
            u_x = (
                u_x_base
                + 0.2 * np.sin(X + time_factor)
                + 0.05 * np.random.randn(nx, ny)
            )
            u_y = (
                u_y_base
                + 0.2 * np.sin(Y + time_factor)
                + 0.05 * np.random.randn(nx, ny)
            )

            # Generate magnetic potential with smaller magnitude
            A = 2.5e-4 * np.sin(2 * X) * np.sin(
                2 * Y + time_factor
            ) + 5e-5 * np.random.randn(nx, ny)

            # Store in data array
            data[sample, t, :, :, 0] = u_x
            data[sample, t, :, :, 1] = u_y
            data[sample, t, :, :, 2] = A

    # Save the data
    np.save(mini_data_path, data)

    return mini_data_path


@pytest.fixture
def optimizer_config():
    """Return a test configuration for optimizers."""
    return {
        "optimizer_params": {
            "optimizer_type": "adam",
            "lr": 1.0e-3,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
            "eps": 1.0e-8,
            "use_scheduler": True,
            "scheduler_type": "multistep",
            "milestones": [10, 20],
            "gamma": 0.5,
        }
    }


@pytest.fixture
def mini_config(mini_data_path, mini_config_path):
    """Create a test configuration file."""
    # Define the configuration
    config = {
        "model_params": {
            "model_type": "tfno",
            "width": 8,
            "n_layers": 2,
            "modes1": 8,
            "modes2": 8,
            "modes3": 8,
            "fc_dim": 64,
            "in_dim": 3,
            "out_dim": 3,
            "activation": "gelu",
            "pad_x": 0.05,
            "pad_y": 0.05,
            "pad_z": 0.00,
            "rank": 1.00,
            "factorization": "tucker",
        },
        "normalization_params": {
            "type": "physics",
            "input_norm": [1.0, 1.0, 2.5e-4],
            "output_norm": [1.0, 1.0, 2.5e-4],
        },
        "dataset_params": {
            "data_dir": os.path.dirname(mini_data_path) + "/",
            "data_file": os.path.basename(mini_data_path),
            "name": "MHD_TFNO3d_test",
            "num": 10,
            "num_train": 8,
            "num_test": 2,
            "seed": 42,
        },
        "loss_params": {"type": "mse", "reduction": "mean"},
        "train_loader_params": {
            "batch_size": 2,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
        },
        "val_loader_params": {
            "batch_size": 2,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": False,
        },
        "test_loader_params": {
            "batch_size": 2,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": False,
        },
        "optimizer_params": {
            "optimizer_type": "adam",  # Add optimizer type
            "betas": [0.9, 0.999],
            "lr": 1.0e-3,
            "weight_decay": 0.0,  # Add weight decay
            "scheduler_type": "multistep",  # Add scheduler type
            "milestones": [10, 20],
            "gamma": 0.5,
            "use_scheduler": True,
        },
        "train_params": {
            "epochs": 3,
            "ckpt_freq": 1,
            "ckpt_path": "checkpoints/test/model",
        },
    }

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(mini_config_path), exist_ok=True)

    # Write the configuration to a file
    with open(mini_config_path, "w") as f:
        yaml.dump(config, f)

    return config
