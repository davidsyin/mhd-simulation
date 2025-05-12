"""Tests for diffusion model integration and factory."""

import pytest
import torch

from src.diffusion.models.backbones.unet import UNet
from src.diffusion.models.elucidated_diffusion import ElucidatedDiffusion
from src.diffusion.models.diffusion_factory import (
    create_diffusion_model,
    register_diffusion_model,
    DIFFUSION_MODEL_REGISTRY,
    _register_all_diffusion_models,
)


@pytest.fixture
def device():
    """Return the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_unet(device):
    """Return a small UNet for testing."""
    return UNet(dim=32, channels=3, dim_mults=(1, 2), dropout=0.0).to(device)


@pytest.fixture
def small_diffusion_model(small_unet, device):
    """Return a small ElucidatedDiffusion model for testing."""
    return ElucidatedDiffusion(
        small_unet,
        image_size=32,
        channels=3,
        num_sample_steps=4,  # Use small number of steps for testing
        sigma_min=0.002,
        sigma_max=80,
        sigma_data=0.5,
        rho=7,
    ).to(device)


@pytest.fixture
def sample_images(device):
    """Create sample images for testing."""
    batch_size = 2
    channels = 3
    image_size = 32
    return torch.randn(batch_size, channels, image_size, image_size).to(device)


def test_register_diffusion_model():
    """Test registering a diffusion model."""
    # Clear or store existing registry to restore later
    original_registry = DIFFUSION_MODEL_REGISTRY.copy()

    try:
        # Register a test model constructor
        @register_diffusion_model("test_model")
        def create_test_model(model_params):
            # Return a simple UNet as a test
            return UNet(
                dim=model_params.get("dim", 32),
                channels=model_params.get("channels", 3),
            )

        # Check if it was registered correctly
        assert "test_model" in DIFFUSION_MODEL_REGISTRY
        assert DIFFUSION_MODEL_REGISTRY["test_model"] == create_test_model

        # Test with variant
        @register_diffusion_model("test_model", "variant")
        def create_test_model_variant(model_params):
            return UNet(
                dim=model_params.get("dim", 64),
                channels=model_params.get("channels", 3),
                self_condition=True,
            )

        assert "test_model_variant" in DIFFUSION_MODEL_REGISTRY

    finally:
        # Restore original registry
        DIFFUSION_MODEL_REGISTRY.clear()
        DIFFUSION_MODEL_REGISTRY.update(original_registry)


def test_create_diffusion_model(device):
    """Test creating a diffusion model using the factory."""
    # Test creating an elucidated diffusion model
    config = {
        "model_params": {
            "model_type": "elucidated",
            "base_dim": 32,
            "dim_mults": (1, 2),
            "channels": 3,
            "image_size": 32,
            "num_sample_steps": 4,
        }
    }

    model = create_diffusion_model(config).to(device)

    # Verify it's an ElucidatedDiffusion model
    assert isinstance(model, ElucidatedDiffusion)
    assert model.channels == 3
    assert model.image_size == 32
    assert model.num_sample_steps == 4

    # Test with non-existent model type
    with pytest.raises(ValueError):
        invalid_config = {"model_params": {"model_type": "nonexistent_model"}}
        create_diffusion_model(invalid_config)


def test_diffusion_integration(device):
    """Test integration of diffusion model components."""
    # Create a small UNet
    unet = UNet(dim=32, channels=3, dim_mults=(1, 2), dropout=0.1).to(device)

    # Create ElucidatedDiffusion model
    diffusion = ElucidatedDiffusion(
        unet,
        image_size=64,
        channels=3,
        num_sample_steps=4,  # Small number for testing
    ).to(device)

    # Create some input data
    batch_size = 2
    channels = 3
    h, w = 64, 64

    # Create inputs
    x = torch.randn(batch_size, channels, h, w).to(device)

    # Forward pass (training)
    loss = diffusion(x)

    # Check loss
    assert torch.isfinite(loss)

    # Backward pass should work
    loss.backward()

    # Make sure all parameters have gradients
    for name, param in diffusion.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"

    # Test minimal sampling
    with torch.no_grad():
        self_cond = x  # Use dummy images as self-conditioning
        samples = diffusion.sample(
            self_cond, batch_size=batch_size, num_sample_steps=2, clamp=True
        )

    # Check shape and range
    assert samples.shape == x.shape
    assert torch.all(samples >= -1.0) and torch.all(samples <= 1.0)


def test_diffusion_with_config(device):
    """Test creating and using a diffusion model from a configuration dictionary."""
    # Create a config dictionary similar to the format used in your project
    config = {
        "model_params": {
            "model_type": "elucidated",
            "base_dim": 32,
            "dim_mults": (1, 2, 4),
            "channels": 3,
            "image_size": 64,
            "num_sample_steps": 20,
            "sigma_min": 0.002,
            "sigma_max": 80,
            "sigma_data": 0.5,
            "rho": 7,
            "self_condition": True,
            "learned_variance": False,
            "flash_attn": False,
        },
        "normalization_params": {
            "type": "physics",
            "input_norm": [1.0, 1.0, 2.5e-4],
            "output_norm": [1.0, 1.0, 2.5e-4],
        },
        "dataset_params": {
            "data_dir": "/path/to/data/",
            "data_file": "mhd_data.npy",
            "name": "MHD_Diffusion",
            "num": 1000,
            "num_train": 950,
            "num_test": 50,
            "seed": 42,
            "sub_t": 1,
            "sub_x": 1,
        },
        "loss_params": {"type": "mse", "reduction": "mean"},
        "train_loader_params": {
            "batch_size": 4,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True,
        },
        "optimizer_params": {
            "optimizer_type": "adamw",
            "lr": 1.0e-4,
            "weight_decay": 1.0e-5,
            "betas": [0.9, 0.999],
            "use_scheduler": True,
            "scheduler_type": "cosine",
            "T_max": 100,
            "eta_min": 1e-6,
        },
        "train_params": {
            "epochs": 100,
            "ckpt_path": "checkpoints/diffusion_model.pt",
        },
    }

    # Create the diffusion model from config
    model = create_diffusion_model(config).to(device)

    # Verify it's the right type
    assert isinstance(model, ElucidatedDiffusion)

    # Check key properties from config
    assert model.channels == config["model_params"]["channels"]
    assert model.image_size == config["model_params"]["image_size"]
    assert model.num_sample_steps == config["model_params"]["num_sample_steps"]
    assert model.sigma_min == config["model_params"]["sigma_min"]
    assert model.sigma_max == config["model_params"]["sigma_max"]
    assert model.self_condition == config["model_params"]["self_condition"]

    # Create input data for forward pass
    batch_size = 2
    x = torch.randn(batch_size, model.channels, model.image_size, model.image_size).to(
        device
    )

    # Verify forward pass (training) works
    loss = model(x)
    assert torch.is_tensor(loss)
    assert loss.dim() == 0  # Should be a scalar loss

    # Test backward pass
    loss.backward()

    # Verify minimal sampling with self-conditioning works
    with torch.no_grad():
        # Use a very small number of steps for testing efficiency
        samples = model.sample(x, batch_size=batch_size, num_sample_steps=2)
        assert samples.shape == x.shape
        assert torch.all(samples >= -1.0) and torch.all(samples <= 1.0)

        # Test DPM++ sampling
        samples_dpmpp = model.sample_using_dpmpp(
            x, batch_size=batch_size, num_sample_steps=2
        )
        assert samples_dpmpp.shape == x.shape
        assert torch.all(samples_dpmpp >= -1.0) and torch.all(samples_dpmpp <= 1.0)


def test_diffusion_registry_consistency():
    """Test that the diffusion model registry contains expected models."""
    # Make sure registry is populated
    from src.diffusion.models.diffusion_factory import (
        _register_all_diffusion_models,
        DIFFUSION_MODEL_REGISTRY,
    )

    # Call registration function to ensure all models are registered
    _register_all_diffusion_models()

    # Check that at least elucidated diffusion is registered
    assert (
        "elucidated" in DIFFUSION_MODEL_REGISTRY
    ), "ElucidatedDiffusion should be registered"

    # Print available models for debugging
    available_models = list(DIFFUSION_MODEL_REGISTRY.keys())
    print(f"Available diffusion models: {available_models}")

    # If more models are added in the future, add assertions for them here
