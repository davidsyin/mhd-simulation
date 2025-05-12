"""Tests for diffusion model architectures."""

import pytest
import torch
import numpy as np

from src.diffusion.models.backbones.unet import UNet
from src.diffusion.models.elucidated_diffusion import ElucidatedDiffusion


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


def test_unet_init(device):
    """Test UNet initialization with various parameters."""
    # Basic initialization
    unet = UNet(dim=64, channels=3, dim_mults=(1, 2, 4)).to(device)

    # Verify some attributes
    assert unet.channels == 3
    assert unet.self_condition is False
    assert hasattr(unet, "init_conv")
    assert hasattr(unet, "downs")
    assert hasattr(unet, "ups")
    assert hasattr(unet, "mid_block1")
    assert hasattr(unet, "mid_block2")
    assert hasattr(unet, "final_res_block")
    assert hasattr(unet, "final_conv")

    # Check number of downsampling/upsampling blocks
    dim_mults = (1, 2, 4)
    assert len(unet.downs) == len(
        dim_mults
    )  # UNet creates one down block per multiplier
    assert len(unet.ups) == len(dim_mults)  # And the same number of up blocks

    # Check learned variance
    unet_var = UNet(dim=64, channels=3, learned_variance=True).to(device)

    # Output dimension should be 2x channels when learned_variance is True
    assert unet_var.out_dim == 6

    # Check self-conditioning
    unet_self = UNet(dim=64, channels=3, self_condition=True).to(device)

    assert unet_self.self_condition is True


def test_unet_forward(device):
    """Test UNet forward pass."""
    batch_size, channels, h, w = 2, 3, 32, 32

    # Create a UNet with small dimensions for testing
    unet = UNet(dim=32, channels=channels, dim_mults=(1, 2)).to(device)

    # Create input
    x = torch.randn(batch_size, channels, h, w).to(device)
    t = torch.randint(0, 1000, (batch_size,)).float().to(device)

    # Forward pass
    output = unet(x, t)

    # Check output shape
    assert output.shape == (batch_size, channels, h, w)

    # Test with self-conditioning
    unet_self = UNet(dim=32, channels=channels, self_condition=True).to(device)

    # First pass without self-conditioning
    output1 = unet_self(x, t)
    # Second pass with self-conditioning
    output2 = unet_self(x, t, x_self_cond=output1)

    # Check output shapes
    assert output1.shape == output2.shape == (batch_size, channels, h, w)
    # Outputs should be different with self-conditioning
    assert not torch.allclose(output1, output2)

    # Test with learned variance
    unet_var = UNet(dim=32, channels=channels, learned_variance=True).to(device)

    output_var = unet_var(x, t)
    # Output should have 2x channels
    assert output_var.shape == (batch_size, channels * 2, h, w)


def test_unet_downsample_factor(device):
    """Test the UNet's downsample_factor property."""
    # Create UNets with different numbers of downsampling layers
    unet1 = UNet(dim=32, channels=3, dim_mults=(1, 2)).to(device)
    unet2 = UNet(dim=32, channels=3, dim_mults=(1, 2, 4)).to(device)
    unet3 = UNet(dim=32, channels=3, dim_mults=(1, 2, 4, 8)).to(device)

    # Check downsample factors
    assert unet1.downsample_factor == 2  # One downsampling layer (2^1)
    assert unet2.downsample_factor == 4  # Two downsampling layers (2^2)
    assert unet3.downsample_factor == 8  # Three downsampling layers (2^3)


def test_elucidated_diffusion_init(small_unet, device):
    """Test ElucidatedDiffusion initialization."""
    model = ElucidatedDiffusion(
        small_unet, image_size=32, channels=3, num_sample_steps=32
    ).to(device)

    assert model.image_size == 32
    assert model.channels == 3
    assert model.num_sample_steps == 32
    assert hasattr(model, "net")
    assert hasattr(model, "sigma_min")
    assert hasattr(model, "sigma_max")
    assert hasattr(model, "sigma_data")
    assert model.self_condition == small_unet.self_condition


def test_elucidated_diffusion_coefficients(small_diffusion_model):
    """Test the coefficient functions of ElucidatedDiffusion."""
    model = small_diffusion_model

    # Test with scalar value
    sigma_scalar = 0.5
    batch_size = 2

    # Convert scalar to tensor for testing
    sigma_tensor = torch.tensor([sigma_scalar] * batch_size).to(model.device)

    # Test functions with tensor input
    c_skip = model.c_skip(sigma_tensor)
    c_out = model.c_out(sigma_tensor)
    c_in = model.c_in(sigma_tensor)
    c_noise = model.c_noise(sigma_tensor)

    assert torch.is_tensor(c_skip)
    assert torch.is_tensor(c_out)
    assert torch.is_tensor(c_in)
    assert torch.is_tensor(c_noise)

    # Test behavior as sigma → 0 and sigma → ∞
    sigma_small = torch.tensor([0.001] * batch_size).to(model.device)
    sigma_large = torch.tensor([1000.0] * batch_size).to(model.device)

    # As sigma → 0, c_skip → 1
    assert model.c_skip(sigma_small).mean().item() > 0.99
    # As sigma → ∞, c_skip → 0
    assert model.c_skip(sigma_large).mean().item() < 0.01


def test_noise_distribution(small_diffusion_model):
    """Test the noise distribution sampling."""
    model = small_diffusion_model
    batch_size = 1000

    # Sample noise levels
    sigmas = model.noise_distribution(batch_size)

    # Check shape
    assert sigmas.shape == (batch_size,)

    # Check range - should be positive
    assert torch.all(sigmas > 0)

    # Statistical test - check general properties rather than exact values
    assert sigmas.mean().item() > 0
    assert sigmas.std().item() > 0


def test_preconditioned_network_forward(small_diffusion_model, sample_images):
    """Test the preconditioned network forward pass."""
    model = small_diffusion_model
    images = sample_images
    batch_size = images.shape[0]

    # Test with tensor sigma
    sigma_tensor = torch.ones(batch_size).to(model.device) * 0.5
    output_tensor = model.preconditioned_network_forward(images, sigma_tensor)
    assert output_tensor.shape == images.shape

    # Test with clamping
    output_clamp = model.preconditioned_network_forward(
        images, sigma_tensor, clamp=True
    )
    assert output_clamp.shape == images.shape
    assert torch.all(output_clamp >= -1.0) and torch.all(output_clamp <= 1.0)


def test_loss_weight(small_diffusion_model):
    """Test the loss weighting function."""
    model = small_diffusion_model
    batch_size = 4
    sigmas = torch.tensor([0.1, 0.5, 1.0, 10.0]).to(model.device)

    weights = model.loss_weight(sigmas)

    # Check shape
    assert weights.shape == sigmas.shape

    # Weights should be higher for smaller sigmas
    assert weights[0] > weights[1] > weights[2] > weights[3]


def test_elucidated_diffusion_forward(small_diffusion_model, sample_images):
    """Test the forward pass (training) of ElucidatedDiffusion."""
    model = small_diffusion_model
    images = sample_images

    # Forward pass (simulating training)
    loss = model(images)

    # Check loss is a scalar
    assert loss.dim() == 0
    assert torch.isfinite(loss)

    # Backward pass should work
    loss.backward()

    # All parameters should have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
