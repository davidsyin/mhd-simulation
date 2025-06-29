"""Tests for the model modules."""

import pytest
import torch
from src.neurops.model_factory import create_model, MODEL_REGISTRY
from src.neurops.tfno_neuralop import TFNO_neuralop
from src.neurops.tfno import TFNO
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
    from src.data.neurops_dataset import get_dataloaders

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
        train_size=8,
        val_plus_test_size=2,
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
    from src.neurops.tfno import TFNO

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
    from src.neurops.tfno import TFNO

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


def test_fno_model_creation(device):
    """Test FNO model creation and structure."""
    # Create config for 3D FNO model
    config = {
        "model_params": {
            "model_type": "fno",
            "model_variant": "3d",
            "in_channels": 6,
            "out_channels": 3,
            "dimension": 3,
            "latent_channels": 16,
            "num_fno_layers": 3,
            "num_fno_modes": 8,
            "padding": 0,
            "activation_fn": "gelu",
            "coord_features": False,
        }
    }

    # Create model
    model = create_model(config).to(device)

    # Verify model type
    assert model.__class__.__name__ == "FNO", "Model should be an FNO instance"

    # Test model attributes
    assert hasattr(model, "forward"), "Model should have forward method"

    # Check that the model has expected attributes
    assert hasattr(model, "decoder_net"), "FNO should have decoder_net attribute"


def test_fno_model_forward(device):
    """Test FNO model forward pass with correct shapes."""
    # Create minimal config for the FNO model
    config = {
        "model_params": {
            "model_type": "fno",
            "model_variant": "3d",
            "in_channels": 6,  # 3 for grid coordinates and 3 for data
            "out_channels": 3,
            "dimension": 3,
            "latent_channels": 16,
            "num_fno_layers": 2,
            "num_fno_modes": 4,
            "padding": 0,
            "activation_fn": "gelu",
            "coord_features": False,
        }
    }

    # Create model
    model = create_model(config).to(device)

    # Create dummy input tensor with shape (batch, channels, t, x, y)
    batch_size = 2
    in_channels = 6  # 3 for grid coordinates and 3 for data
    t_size = 10  # Time steps
    x_size, y_size = 16, 16  # Spatial dimensions

    # Create input tensor
    x = torch.randn(batch_size, in_channels, t_size, x_size, y_size).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    expected_shape = (batch_size, 3, t_size, x_size, y_size)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {output.shape}"
    )


def test_tfno_different_factorizations(device):
    """Test different factorization methods in TFNO model."""
    factorizations = ["cp", "tucker"]  # Can also test "tt" if supported

    for factorization in factorizations:
        # Create config with the current factorization
        config = {
            "model_params": {
                "model_type": "tfno",
                "model_variant": "3d",
                "in_channels": 6,
                "out_channels": 3,
                "dimension": 3,
                "latent_channels": 8,
                "num_fno_layers": 2,
                "num_fno_modes": 4,
                "padding": 4,
                "rank": 0.5,
                "factorization": factorization,
            }
        }

        # Create model
        model = create_model(config).to(device)

        # Create dummy input
        x = torch.randn(1, 6, 8, 8, 8).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Check output shape
        assert output.shape == (
            1,
            3,
            8,
            8,
            8,
        ), f"Wrong output shape for factorization {factorization}"


def test_fno_integration_with_dataloader(generate_mini_mhd_data, device):
    """Test FNO model integration with dataset and dataloader."""
    from src.data.neurops_dataset import get_dataloaders

    # Create config for FNO
    config = {
        "model_params": {
            "model_type": "fno",
            "model_variant": "3d",
            "in_channels": 6,
            "out_channels": 3,
            "dimension": 3,
            "latent_channels": 8,
            "num_fno_layers": 2,
            "num_fno_modes": 4,
            "coord_features": False,
        },
        "normalization_params": {
            "type": "physics",
            "input_norm": [1.0, 1.0, 2.5e-4],
            "output_norm": [1.0, 1.0, 2.5e-4],
        },
    }

    # Create model
    model = create_model(config).to(device)

    # Get dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        generate_mini_mhd_data,
        normalization_config={"normalization_params": config["normalization_params"]},
        batch_size=2,
        num_workers=0,
        train_size=8,
        val_plus_test_size=2,
        seed=42,
        sub_t=1,
        sub_x=1,
    )

    # Test with a batch from dataloader
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Check output shape
    assert output.shape == y.shape, (
        f"Expected output shape {y.shape}, got {output.shape}"
    )

    # Check output is finite
    assert torch.isfinite(output).all(), "Output should have finite values"


def test_model_parameters_and_training(device):
    """Test model parameters and a simple training step."""
    # Create a simple model config
    config = {
        "model_params": {
            "model_type": "fno",  # Test with FNO as it's the preferred model
            "model_variant": "3d",
            "in_channels": 6,
            "out_channels": 3,
            "dimension": 3,
            "latent_channels": 8,
            "num_fno_layers": 2,
            "num_fno_modes": 4,
            "padding": 0,
        },
    }

    # Create model
    model = create_model(config).to(device)

    # Check that model has parameters
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0, "Model should have parameters"

    # Test gradient calculation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Create dummy input and target
    x = torch.randn(2, 6, 10, 16, 16).to(device)
    y = torch.randn(2, 3, 10, 16, 16).to(device)

    # Forward pass
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)

    # Backward pass
    loss.backward()

    # Check that gradients are computed
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
            has_grad = True
            break

    assert has_grad, "At least one parameter should have non-zero gradient"


def test_all_registered_models(device):
    """Test all registered models in the MODEL_REGISTRY."""
    # Basic configuration for testing - use small sizes for efficiency
    base_config = {
        "model_params": {
            # These will be overwritten per model
            "in_channels": 6,
            "out_channels": 3,
            "dimension": 3,
            "latent_channels": 8,
            "num_fno_layers": 2,
            "num_fno_modes": 4,
            "padding": 0,
            "in_dim": 6,
            "out_dim": 3,
            "width": 8,
            "n_layers": 2,
            "modes1": 4,
            "modes2": 4,
            "modes3": 4,
            "rank": 0.5,
            "factorization": "cp",
        }
    }

    # Create small input tensor for testing
    x = torch.randn(1, 6, 8, 8, 8).to(device)

    # Test each registered model
    for model_key in MODEL_REGISTRY:
        print(f"Testing model: {model_key}")

        # Update config with the current model type
        config = base_config.copy()
        config["model_params"] = base_config["model_params"].copy()

        # Split model key into type and variant
        if "_" in model_key:
            model_type, model_variant = model_key.split("_", 1)
            config["model_params"]["model_type"] = model_type
            config["model_params"]["model_variant"] = model_variant
        else:
            config["model_params"]["model_type"] = model_key

        try:
            # Create model
            model = create_model(config).to(device)

            # Forward pass
            with torch.no_grad():
                output = model(x)

            # Check output shape and values
            assert output.shape[0] == 1, (
                f"Output batch size should be 1 for {model_key}"
            )
            assert output.shape[1] == 3, f"Output channels should be 3 for {model_key}"
            assert torch.isfinite(output).all(), (
                f"Output should be finite for {model_key}"
            )

        except Exception as e:
            pytest.fail(f"Model {model_key} failed: {str(e)}")
