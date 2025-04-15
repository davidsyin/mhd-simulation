"""Tests for the model modules."""

import pytest
import torch
from src.models.model_factory import create_model
from src.models.tfno_neuralop import TFNO_neuralop
from src.utils.config import load_config


def test_model_factory(mini_config):
    """Test model factory creates proper model."""
    # Create model from config
    model = create_model(mini_config)

    # Check type
    assert isinstance(model, TFNO_neuralop), (
        "Factory should create TFNO_neuralop model with default config"
    )

    # Check model has expected attributes
    assert hasattr(model, "model"), "TFNO_neuralop should have 'model' attribute"
    assert hasattr(model, "forward"), "TFNO_neuralop should have 'forward' method"


def test_tfno_model_forward(mini_config, device):
    """Test TFNO model forward pass with correct shapes."""
    # Create model from config
    model = create_model(mini_config).to(device)

    # Create dummy input tensor with grid coordinates
    # TFNO3d expects (batch, channels, t, x, y)
    batch_size = 2
    # Input now needs to have 6 channels (3 grid + 3 data)
    in_channels = 6  # 3 for grid coordinates and 3 for data
    out_channels = mini_config["model_params"]["out_dim"]
    t_size = 21  # Time steps
    x_size, y_size = 32, 32  # Spatial dimensions

    # Create input tensor
    x = torch.randn(batch_size, in_channels, t_size, x_size, y_size).to(device)

    # Forward pass
    with torch.no_grad():
        y = model(x)

    # Check output shape - output should have 3 channels (no grid coordinates)
    expected_output_shape = (batch_size, out_channels, t_size, x_size, y_size)
    assert y.shape == expected_output_shape, (
        f"Expected output shape {expected_output_shape}, got {y.shape}"
    )


def test_model_with_actual_data(mini_config_path, generate_mini_mhd_data, device):
    """Test model with actual data from dataloader."""
    from src.data.dataset import get_dataloaders

    # Load config
    config = load_config(mini_config_path)

    # Create model
    model = create_model(config).to(device)

    # Get dataloaders
    train_loader, _, _ = get_dataloaders(
        generate_mini_mhd_data,
        normalization_config={"normalization_params": config["normalization_params"]},
        batch_size=2,
        num_workers=0,
        train_ratio=0.8,
        seed=42,
        sub_t=1,
        sub_x=1,
    )

    # Get a batch
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape matches target shape, not input shape
    # Input has grid embeddings, output doesn't
    assert output.shape == y.shape, (
        f"Expected output shape {y.shape}, got {output.shape}"
    )

    # Check output is not NaN or Inf
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains Inf values"


def test_custom_tfno_model(device):
    """Test the custom TFNO model."""
    from src.models.tfno import TFNO

    # Create minimal config for the custom TFNO model
    config = {
        "model_params": {
            "model_type": "tfno",
            "variant": "3d",
            "in_channels": 6,
            "out_channels": 3,
            "dimension": 3,
            "num_fno_layers": 2,
            "latent_channels": 8,
            "num_fno_modes": 4,
            "padding": 4,
            "rank": 0.5,
            "factorization": "cp",
        }
    }

    # Create model
    model = create_model(config).to(device)

    # Verify model type
    from src.models.tfno import TFNO

    assert isinstance(model, TFNO), (
        "Factory should create custom TFNO model for tfno_3d variant"
    )

    # Test forward pass
    batch_size = 2
    input_shape = (batch_size, 6, 10, 16, 16)
    x = torch.randn(input_shape).to(device)

    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, 3, 10, 16, 16)
    assert output.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {output.shape}"
    )
