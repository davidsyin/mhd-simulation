"""Fixtures specific to testing the diffusion module."""

import pytest
import torch
from src.diffusion.models.backbones.unet import UNet
from src.diffusion.models.elucidated_diffusion import ElucidatedDiffusion


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
