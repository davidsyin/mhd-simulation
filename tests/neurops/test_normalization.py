"""Tests for the normalization modules."""

import pytest
import torch
import numpy as np
from src.normalizations import create_normalization
from src.normalizations.standard_norm import StandardNormalization
from src.normalizations.min_max_norm import MinMaxNormalization
from src.normalizations.physics_norm import PhysicsNormalization
from src.normalizations.identity_norm import IdentityNormalization


@pytest.fixture
def sample_5d_tensor():
    """Create a sample 5D tensor for testing normalizations."""
    # Shape: (batch=2, channels=3, t=4, x=8, y=8)
    torch.manual_seed(42)

    # Create a tensor with different ranges for each channel
    tensor = torch.zeros(2, 3, 4, 8, 8)

    # Channel 0 (u_x): values around [-1, 1]
    tensor[:, 0] = torch.randn(2, 4, 8, 8)

    # Channel 1 (u_y): values around [-1, 1]
    tensor[:, 1] = torch.randn(2, 4, 8, 8)

    # Channel 2 (A): values around 1e-4
    tensor[:, 2] = 2.5e-4 * torch.randn(2, 4, 8, 8)

    return tensor


def test_factory_registration():
    """Test normalization factory registrations."""
    # Try creating each normalization type
    identity_norm = create_normalization({"normalization_params": {"type": "identity"}})
    standard_norm = create_normalization({"normalization_params": {"type": "standard"}})
    minmax_norm = create_normalization({"normalization_params": {"type": "minmax"}})
    physics_norm = create_normalization({"normalization_params": {"type": "physics"}})

    # Check correct types
    assert isinstance(identity_norm, IdentityNormalization)
    assert isinstance(standard_norm, StandardNormalization)
    assert isinstance(minmax_norm, MinMaxNormalization)
    assert isinstance(physics_norm, PhysicsNormalization)


def test_identity_normalization(sample_5d_tensor):
    """Test identity normalization."""
    tensor = sample_5d_tensor

    # Create identity normalization
    norm = create_normalization({"normalization_params": {"type": "identity"}})

    # Apply normalization and denormalization
    normalized = norm.normalize(tensor)
    denormalized = norm.denormalize(normalized)

    # Check that tensor is unchanged
    assert torch.allclose(tensor, normalized), (
        "Identity normalization should not change the tensor"
    )
    assert torch.allclose(tensor, denormalized), (
        "Identity denormalization should not change the tensor"
    )


def test_standard_normalization(sample_5d_tensor):
    """Test standard normalization."""
    tensor = sample_5d_tensor

    # Calculate means and stds for each channel
    means = tensor.mean(dim=(0, 2, 3, 4)).tolist()
    stds = tensor.std(dim=(0, 2, 3, 4)).tolist()

    # Create standard normalization
    norm = create_normalization(
        {"normalization_params": {"type": "standard", "mean": means, "std": stds}}
    )

    # Apply normalization
    normalized = norm.normalize(tensor)

    # Check that normalized data has approximately zero mean and unit variance
    norm_means = normalized.mean(dim=(0, 2, 3, 4))
    norm_stds = normalized.std(dim=(0, 2, 3, 4))

    assert torch.allclose(norm_means, torch.zeros_like(norm_means), atol=1e-5), (
        "Normalized data should have zero mean"
    )
    assert torch.allclose(norm_stds, torch.ones_like(norm_stds), atol=1e-4), (
        "Normalized data should have unit variance"
    )

    # Check denormalization
    denormalized = norm.denormalize(normalized)
    assert torch.allclose(tensor, denormalized, atol=1e-5), (
        "Denormalized data should match original data"
    )


def test_minmax_normalization(sample_5d_tensor):
    """Test min-max normalization."""
    tensor = sample_5d_tensor

    # Calculate min and max for each channel
    min_vals = tensor.min(dim=0)[0].min(dim=1)[0].min(dim=1)[0].min(dim=1)[0].tolist()
    max_vals = tensor.max(dim=0)[0].max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].tolist()

    # Create min-max normalization with range [0, 1]
    norm = create_normalization(
        {
            "normalization_params": {
                "type": "minmax",
                "min_val": min_vals,
                "max_val": max_vals,
                "feature_range": [0, 1],
            }
        }
    )

    # Apply normalization
    normalized = norm.normalize(tensor)

    # Check that normalized data is in [0, 1] range
    assert torch.all(normalized >= 0) and torch.all(normalized <= 1), (
        "Normalized data should be in [0, 1] range"
    )

    # Check denormalization
    denormalized = norm.denormalize(normalized)
    assert torch.allclose(tensor, denormalized, atol=1e-5), (
        "Denormalized data should match original data"
    )


def test_physics_normalization(sample_5d_tensor):
    """Test physics-aware normalization."""
    tensor = sample_5d_tensor

    # Typical scales for MHD quantities
    input_norm = [1.0, 1.0, 2.5e-4]  # u_x, u_y, A scales

    # Create physics normalization
    norm = create_normalization(
        {
            "normalization_params": {
                "type": "physics",
                "input_norm": input_norm,
                "output_norm": input_norm,
            }
        }
    )

    # Apply normalization
    normalized = norm.normalize(tensor)

    # Check scaling - each channel should be divided by its scale
    for i, scale in enumerate(input_norm):
        orig_channel = tensor[:, i : i + 1]
        norm_channel = normalized[:, i : i + 1]
        expected = orig_channel / scale
        assert torch.allclose(norm_channel, expected, atol=1e-5), (
            f"Channel {i} should be scaled by {scale}"
        )

    # Check denormalization
    denormalized = norm.denormalize(normalized)
    assert torch.allclose(tensor, denormalized, atol=1e-5), (
        "Denormalized data should match original data"
    )
