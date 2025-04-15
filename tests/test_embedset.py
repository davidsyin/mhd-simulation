"""Tests for the EmbeddedMHDDataset module."""

import pytest
import torch
import numpy as np
from src.data.embedset import EmbeddedMHDDataset


def test_embedded_mhd_dataset(mini_config):
    """Test EmbeddedMHDDataset initialization and basic functionality."""
    # Create dummy data
    batch_size = 5
    nx, ny = 16, 16  # Spatial dimensions
    nt = 10  # Time steps

    # Create input with shape [batch, channels, 1, nx, ny]
    data_tensor = torch.rand(batch_size, 3, 1, nx, ny)

    # Create target with shape [batch, channels, nt, nx, ny]
    target_tensor = torch.rand(batch_size, 3, nt, nx, ny)

    # Create dataset
    dataset = EmbeddedMHDDataset(
        data_tensor,
        target_tensor,
        nx=nx,
        ny=ny,
        normalization_config={
            "normalization_params": mini_config["normalization_params"]
        },
        model_config=mini_config,
    )

    # Check dataset length
    assert len(dataset) == batch_size, f"Dataset length should be {batch_size}"

    # Test __getitem__
    x, y = dataset[0]

    # Check if the returned shapes are correct
    # The implementation returns x_embedded (6 channels) when positional_embedding is None or False
    # and returns x_expanded (3 channels) when positional_embedding is a string or other non-False value
    if mini_config["model_params"].get("positional_embedding"):
        expected_x_shape = (
            3,
            nt,
            nx,
            ny,
        )  # Original data only (when positional_embedding is specified)
    else:
        expected_x_shape = (
            6,
            nt,
            nx,
            ny,
        )  # 3 grid (t,x,y) + 3 data channels (when None or False)

    expected_y_shape = (3, nt, nx, ny)  # Output should be unchanged

    assert (
        x.shape == expected_x_shape
    ), f"Expected x shape {expected_x_shape}, got {x.shape}"
    assert (
        y.shape == expected_y_shape
    ), f"Expected y shape {expected_y_shape}, got {y.shape}"

    # Test normalization (assuming mini_config has normalization params)
    # We would expect normalized data to typically be in a bounded range
    assert torch.all(torch.isfinite(x)), "Normalized input should have finite values"
    assert torch.all(torch.isfinite(y)), "Normalized target should have finite values"


def test_embedded_mhd_dataset_without_model_config():
    """Test EmbeddedMHDDataset without model_config."""
    # Create dummy data
    batch_size = 3
    nx, ny = 8, 8
    nt = 5

    data_tensor = torch.rand(batch_size, 3, 1, nx, ny)
    target_tensor = torch.rand(batch_size, 3, nt, nx, ny)

    # Create dataset without model_config
    dataset = EmbeddedMHDDataset(data_tensor, target_tensor, nx=nx, ny=ny)

    # Test __getitem__ which should always return embedded data when no model_config
    x, y = dataset[0]

    expected_x_shape = (6, nt, nx, ny)  # Always 6 channels when no model config
    expected_y_shape = (3, nt, nx, ny)

    assert (
        x.shape == expected_x_shape
    ), f"Expected x shape {expected_x_shape}, got {x.shape}"
    assert (
        y.shape == expected_y_shape
    ), f"Expected y shape {expected_y_shape}, got {y.shape}"


def test_embedded_mhd_dataset_with_positional_embedding_string(mini_config):
    """Test EmbeddedMHDDataset with a positional embedding specified as string."""
    # Create dummy data
    batch_size = 3
    nx, ny = 8, 8
    nt = 5

    data_tensor = torch.rand(batch_size, 3, 1, nx, ny)
    target_tensor = torch.rand(batch_size, 3, nt, nx, ny)

    # Create modified config with positional embedding
    test_config = mini_config.copy()
    test_config["model_params"] = mini_config["model_params"].copy()
    test_config["model_params"][
        "positional_embedding"
    ] = "grid"  # Set to an actual type

    # Create dataset with model_config specifying a positional embedding
    dataset = EmbeddedMHDDataset(
        data_tensor,
        target_tensor,
        nx=nx,
        ny=ny,
        normalization_config={
            "normalization_params": mini_config["normalization_params"]
        },
        model_config=test_config,
    )

    # Test __getitem__
    x, y = dataset[0]

    # With positional_embedding set to a string, it should return x_expanded (3 channels)
    expected_x_shape = (
        3,
        nt,
        nx,
        ny,
    )  # Original data when positional_embedding specified
    expected_y_shape = (3, nt, nx, ny)

    assert (
        x.shape == expected_x_shape
    ), f"Expected x shape {expected_x_shape}, got {x.shape}"
    assert (
        y.shape == expected_y_shape
    ), f"Expected y shape {expected_y_shape}, got {y.shape}"
