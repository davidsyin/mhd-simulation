"""Tests for diffusion model sampling."""

import pytest
import torch

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


def test_sample_schedule(small_diffusion_model):
    """Test the sampling schedule generation."""
    model = small_diffusion_model

    # Test with default num_sample_steps
    sigmas = model.sample_schedule()
    assert (
        sigmas.shape[0] == model.num_sample_steps + 1
    )  # +1 for final step with sigma=0

    # Check first and last values
    assert sigmas[0].item() == pytest.approx(model.sigma_max, rel=1e-5)
    assert sigmas[-1].item() == 0.0

    # Test with custom steps
    custom_steps = 10
    sigmas_custom = model.sample_schedule(custom_steps)
    assert sigmas_custom.shape[0] == custom_steps + 1


def test_minimal_sampling(small_diffusion_model, sample_images):
    """Test minimal sampling to ensure it runs without errors."""
    model = small_diffusion_model
    self_cond = sample_images  # Use dummy images as self-conditioning
    batch_size = self_cond.shape[0]

    with torch.no_grad():
        # Use a very small number of steps
        samples = model.sample(
            self_cond, batch_size=batch_size, num_sample_steps=2, clamp=True
        )

    # Check shape
    assert samples.shape == self_cond.shape

    # Check range
    assert torch.all(samples >= -1.0) and torch.all(samples <= 1.0)


def test_minimal_sampling_dpmpp(small_diffusion_model, sample_images):
    """Test minimal DPM++ sampling to ensure it runs without errors."""
    model = small_diffusion_model
    self_cond = sample_images  # Use dummy images as self-conditioning
    batch_size = self_cond.shape[0]

    with torch.no_grad():
        # Use a very small number of steps
        samples = model.sample_using_dpmpp(
            self_cond, batch_size=batch_size, num_sample_steps=2
        )

    # Check shape
    assert samples.shape == self_cond.shape

    # Check range
    assert torch.all(samples >= -1.0) and torch.all(samples <= 1.0)
