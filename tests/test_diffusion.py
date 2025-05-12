"""Tests for the diffusion module."""

import pytest
import torch
import numpy as np
from functools import partial
from math import sqrt

from src.diffusion.utils.utils import exists, default, once, cast_tuple, divisible_by, log
from src.diffusion.models.backbones.unet import UNet
from src.diffusion.models.components.attention import Attention, LinearAttention, Attend
from src.diffusion.models.components.blocks import Block, ResnetBlock
from src.diffusion.models.components.layers import (
    Upsample, Downsample, SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb
)
from src.diffusion.models.components.normalization import RMSNorm
from src.diffusion.models.diffusion_factory import (
    create_diffusion_model, register_diffusion_model, DIFFUSION_MODEL_REGISTRY
)
from src.diffusion.models.elucidated_diffusion import ElucidatedDiffusion


@pytest.fixture
def device():
    """Return the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_unet(device):
    """Return a small UNet for testing."""
    return UNet(
        dim=32,
        channels=3,
        dim_mults=(1, 2),
        dropout=0.0
    ).to(device)


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


# ---------------------- Test Utility Functions ----------------------

def test_exists():
    """Test the exists utility function."""
    assert exists(1) is True
    assert exists(0) is True
    assert exists([]) is True
    assert exists("") is True
    assert exists(None) is False


def test_default():
    """Test the default utility function."""
    assert default(None, 1) == 1
    assert default(0, 1) == 0
    assert default("", "default") == ""
    assert default(None, lambda: "computed") == "computed"
    
    # Test with callable default
    counter = [0]
    def increment():
        counter[0] += 1
        return counter[0]
    
    # Should call the function only when val is None
    assert default(None, increment) == 1
    assert default(None, increment) == 2
    assert default("value", increment) == "value"
    assert counter[0] == 2  # Function should have been called twice


def test_once():
    """Test the once decorator utility function."""
    counter = [0]
    
    @once
    def increment_once(x):
        counter[0] += 1
        return counter[0]
    
    # First call should work
    assert increment_once(1) == 1
    assert counter[0] == 1
    
    # Subsequent calls should do nothing
    assert increment_once(2) is None
    assert increment_once(3) is None
    assert counter[0] == 1  # Counter shouldn't have increased


def test_cast_tuple():
    """Test the cast_tuple utility function."""
    assert cast_tuple(1, 3) == (1, 1, 1)
    assert cast_tuple((1, 2), 3) == (1, 2)
    assert cast_tuple((1, 2, 3), 3) == (1, 2, 3)
    assert cast_tuple([1, 2], 3) == ([1, 2],) * 3  # Lists remain as is, but repeated


def test_divisible_by():
    """Test the divisible_by utility function."""
    assert divisible_by(10, 2) is True
    assert divisible_by(10, 3) is False
    assert divisible_by(0, 5) is True
    

def test_log():
    """Test the log utility function."""
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.log(x)
    assert torch.allclose(log(x), expected)
    
    # Test with small values
    x_small = torch.tensor([1e-30, 1e-100])
    # Should not produce -inf
    result = log(x_small)
    assert not torch.isinf(result).any()


# ---------------------- Test Components ----------------------

def test_rms_norm(device):
    """Test the RMSNorm module."""
    batch_size, channels, h, w = 2, 4, 8, 8
    x = torch.randn(batch_size, channels, h, w).to(device)
    
    norm = RMSNorm(channels).to(device)
    result = norm(x)
    
    # Check output shape
    assert result.shape == x.shape
    
    # Check normalization effect - standard deviation should be approximately 1.0
    # The actual behavior of RMSNorm in this codebase produces std around 1.0, not sqrt(dim)
    channel_std = result.std(dim=(0, 2, 3))
    assert torch.all(channel_std > 0.5) and torch.all(channel_std < 1.5)


def test_block(device):
    """Test the basic convolutional Block."""
    batch_size, in_channels, out_channels, h, w = 2, 4, 8, 16, 16
    x = torch.randn(batch_size, in_channels, h, w).to(device)
    
    block = Block(in_channels, out_channels, dropout=0.1).to(device)
    result = block(x)
    
    # Check output shape
    assert result.shape == (batch_size, out_channels, h, w)
    
    # Test with scale_shift parameter
    scale = torch.ones(batch_size, out_channels, 1, 1).to(device)
    shift = torch.zeros(batch_size, out_channels, 1, 1).to(device)
    
    result_with_scale_shift = block(x, (scale, shift))
    
    # Should produce different output with scale_shift
    assert not torch.allclose(result, result_with_scale_shift)


def test_resnet_block(device):
    """Test the ResnetBlock module."""
    batch_size, in_channels, out_channels, h, w = 2, 4, 8, 16, 16
    x = torch.randn(batch_size, in_channels, h, w).to(device)
    time_emb_dim = 16
    time_emb = torch.randn(batch_size, time_emb_dim).to(device)
    
    # Test with same in/out dims
    block1 = ResnetBlock(in_channels, in_channels).to(device)
    result1 = block1(x)
    assert result1.shape == x.shape
    
    # Test with different in/out dims
    block2 = ResnetBlock(in_channels, out_channels, time_emb_dim=time_emb_dim).to(device)
    result2 = block2(x, time_emb)
    assert result2.shape == (batch_size, out_channels, h, w)


def test_sinusoidal_pos_emb(device):
    """Test the SinusoidalPosEmb module."""
    batch_size = 2
    dim = 32
    
    # Create a batch of time positions
    t = torch.tensor([1.0, 1000.0]).to(device)
    
    embedder = SinusoidalPosEmb(dim).to(device)
    emb = embedder(t)
    
    # Check output shape
    assert emb.shape == (batch_size, dim)
    
    # Test against direct computation
    half_dim = dim // 2
    emb_log = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb_log = torch.exp(torch.arange(half_dim, device=device) * -emb_log)
    
    # Manual computation for one element
    expected_first_half = torch.sin(t[0] * emb_log)
    expected_second_half = torch.cos(t[0] * emb_log)
    
    # Check the first element's embedding
    assert torch.allclose(emb[0, :half_dim], expected_first_half, rtol=1e-5)
    assert torch.allclose(emb[0, half_dim:], expected_second_half, rtol=1e-5)


def test_random_learned_sinusoidal_pos_emb(device):
    """Test the RandomOrLearnedSinusoidalPosEmb module."""
    batch_size = 2
    dim = 32
    
    # Create a batch of time positions
    t = torch.tensor([1.0, 1000.0]).to(device)
    
    # Test random embeddings
    random_embedder = RandomOrLearnedSinusoidalPosEmb(dim, is_random=True).to(device)
    random_emb = random_embedder(t)
    
    # Output should include the original input as first component
    assert random_emb.shape == (batch_size, dim + 1)
    assert torch.allclose(random_emb[:, 0], t)
    
    # Test learned embeddings
    learned_embedder = RandomOrLearnedSinusoidalPosEmb(dim, is_random=False).to(device)
    learned_emb = learned_embedder(t)
    
    assert learned_emb.shape == (batch_size, dim + 1)
    assert torch.allclose(learned_emb[:, 0], t)
    
    # Random and learned should produce different outputs
    # Copy the learned embedder weights to the random one
    with torch.no_grad():
        random_embedder.weights.copy_(learned_embedder.weights)
    
    # Outputs should be the same when weights are identical
    random_emb_new = random_embedder(t)
    learned_emb_new = learned_embedder(t)
    assert torch.allclose(random_emb_new, learned_emb_new)


def test_upsample_downsample(device):
    """Test the Upsample and Downsample layers."""
    batch_size, channels, h, w = 2, 4, 16, 16
    x = torch.randn(batch_size, channels, h, w).to(device)
    
    # Test Upsample with same channels
    upsample = Upsample(channels).to(device)
    up_result = upsample(x)
    assert up_result.shape == (batch_size, channels, h*2, w*2)
    
    # Test Upsample with different output channels
    upsample_diff = Upsample(channels, dim_out=channels*2).to(device)
    up_diff_result = upsample_diff(x)
    assert up_diff_result.shape == (batch_size, channels*2, h*2, w*2)
    
    # Test Downsample with same channels
    downsample = Downsample(channels).to(device)
    down_result = downsample(x)
    assert down_result.shape == (batch_size, channels, h//2, w//2)
    
    # Test Downsample with different output channels
    downsample_diff = Downsample(channels, dim_out=channels*2).to(device)
    down_diff_result = downsample_diff(x)
    assert down_diff_result.shape == (batch_size, channels*2, h//2, w//2)


def test_attend(device):
    """Test the Attend module."""
    batch_size, heads, seq_len, dim_head = 2, 4, 16, 32
    q = torch.randn(batch_size, heads, seq_len, dim_head).to(device)
    k = torch.randn(batch_size, heads, seq_len, dim_head).to(device)
    v = torch.randn(batch_size, heads, seq_len, dim_head).to(device)
    
    # Test with default settings
    attend = Attend().to(device)
    output = attend(q, k, v)
    
    # Check output shape
    assert output.shape == (batch_size, heads, seq_len, dim_head)
    
    # Test with dropout
    attend_dropout = Attend(dropout=0.1).to(device)
    output_dropout = attend_dropout(q, k, v)
    assert output_dropout.shape == (batch_size, heads, seq_len, dim_head)
    
    # Test with custom scale
    attend_scale = Attend(scale=2.0).to(device)
    output_scale = attend_scale(q, k, v)
    assert output_scale.shape == (batch_size, heads, seq_len, dim_head)


def test_attention(device):
    """Test the Attention module."""
    batch_size, channels, h, w = 2, 32, 16, 16
    x = torch.randn(batch_size, channels, h, w).to(device)
    
    # Standard attention with 4 heads
    attention = Attention(
        dim=channels,
        heads=4,
        dim_head=32,
        num_mem_kv=4
    ).to(device)
    
    output = attention(x)
    
    # Check output shape
    assert output.shape == x.shape
    
    # Test with flash attention if available
    attention_flash = Attention(
        dim=channels,
        heads=4,
        dim_head=32,
        num_mem_kv=4,
        flash=True
    ).to(device)
    
    output_flash = attention_flash(x)
    assert output_flash.shape == x.shape


def test_linear_attention(device):
    """Test the LinearAttention module."""
    batch_size, channels, h, w = 2, 32, 16, 16
    x = torch.randn(batch_size, channels, h, w).to(device)
    
    # Linear attention
    linear_attention = LinearAttention(
        dim=channels,
        heads=4,
        dim_head=32,
        num_mem_kv=4
    ).to(device)
    
    output = linear_attention(x)
    
    # Check output shape
    assert output.shape == x.shape


# ---------------------- Test UNet Architecture ----------------------

def test_unet_init(device):
    """Test UNet initialization with various parameters."""
    # Basic initialization
    unet = UNet(
        dim=64,
        channels=3,
        dim_mults=(1, 2, 4)
    ).to(device)
    
    # Verify some attributes
    assert unet.channels == 3
    assert unet.self_condition is False
    assert hasattr(unet, 'init_conv')
    assert hasattr(unet, 'downs')
    assert hasattr(unet, 'ups')
    assert hasattr(unet, 'mid_block1')
    assert hasattr(unet, 'mid_block2')
    assert hasattr(unet, 'final_res_block')
    assert hasattr(unet, 'final_conv')
    
    # Check number of downsampling/upsampling blocks
    dim_mults = (1, 2, 4)
    assert len(unet.downs) == len(dim_mults)  # UNet creates one down block per multiplier
    assert len(unet.ups) == len(dim_mults)    # And the same number of up blocks
    
    # Check learned variance
    unet_var = UNet(
        dim=64,
        channels=3,
        learned_variance=True
    ).to(device)
    
    # Output dimension should be 2x channels when learned_variance is True
    assert unet_var.out_dim == 6
    
    # Check self-conditioning
    unet_self = UNet(
        dim=64,
        channels=3,
        self_condition=True
    ).to(device)
    
    assert unet_self.self_condition is True

def test_unet_forward(device):
    """Test UNet forward pass."""
    batch_size, channels, h, w = 2, 3, 32, 32
    
    # Create a UNet with small dimensions for testing
    unet = UNet(
        dim=32,
        channels=channels,
        dim_mults=(1, 2)
    ).to(device)
    
    # Create input
    x = torch.randn(batch_size, channels, h, w).to(device)
    t = torch.randint(0, 1000, (batch_size,)).float().to(device)
    
    # Forward pass
    output = unet(x, t)
    
    # Check output shape
    assert output.shape == (batch_size, channels, h, w)
    
    # Test with self-conditioning
    unet_self = UNet(
        dim=32,
        channels=channels,
        self_condition=True
    ).to(device)
    
    # First pass without self-conditioning
    output1 = unet_self(x, t)
    # Second pass with self-conditioning
    output2 = unet_self(x, t, x_self_cond=output1)
    
    # Check output shapes
    assert output1.shape == output2.shape == (batch_size, channels, h, w)
    # Outputs should be different with self-conditioning
    assert not torch.allclose(output1, output2)
    
    # Test with learned variance
    unet_var = UNet(
        dim=32,
        channels=channels,
        learned_variance=True
    ).to(device)
    
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


# ---------------------- Test ElucidatedDiffusion ----------------------

def test_elucidated_diffusion_init(small_unet, device):
    """Test ElucidatedDiffusion initialization."""
    model = ElucidatedDiffusion(
        small_unet,
        image_size=32,
        channels=3,
        num_sample_steps=32
    ).to(device)
    
    assert model.image_size == 32
    assert model.channels == 3
    assert model.num_sample_steps == 32
    assert hasattr(model, 'net')
    assert hasattr(model, 'sigma_min')
    assert hasattr(model, 'sigma_max')
    assert hasattr(model, 'sigma_data')
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


def test_sample_schedule(small_diffusion_model):
    """Test the sampling schedule generation."""
    model = small_diffusion_model
    
    # Test with default num_sample_steps
    sigmas = model.sample_schedule()
    assert sigmas.shape[0] == model.num_sample_steps + 1  # +1 for final step with sigma=0
    
    # Check first and last values
    assert sigmas[0].item() == pytest.approx(model.sigma_max, rel=1e-5)
    assert sigmas[-1].item() == 0.0
    
    # Test with custom steps
    custom_steps = 10
    sigmas_custom = model.sample_schedule(custom_steps)
    assert sigmas_custom.shape[0] == custom_steps + 1


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
    output_clamp = model.preconditioned_network_forward(images, sigma_tensor, clamp=True)
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


def test_minimal_sampling(small_diffusion_model, sample_images):
    """Test minimal sampling to ensure it runs without errors."""
    model = small_diffusion_model
    self_cond = sample_images  # Use dummy images as self-conditioning
    batch_size = self_cond.shape[0]
    
    with torch.no_grad():
        # Use a very small number of steps
        samples = model.sample(self_cond, batch_size=batch_size, num_sample_steps=2, clamp=True)
    
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
        samples = model.sample_using_dpmpp(self_cond, batch_size=batch_size, num_sample_steps=2)
    
    # Check shape
    assert samples.shape == self_cond.shape
    
    # Check range
    assert torch.all(samples >= -1.0) and torch.all(samples <= 1.0)


# ---------------------- Test Diffusion Factory ----------------------

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
                channels=model_params.get("channels", 3)
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
                self_condition=True
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
            "num_sample_steps": 4
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
        invalid_config = {
            "model_params": {
                "model_type": "nonexistent_model"
            }
        }
        create_diffusion_model(invalid_config)


# ---------------------- Integration Tests ----------------------

def test_diffusion_integration(device):
    """Test integration of diffusion model components."""
    # Create a small UNet
    unet = UNet(
        dim=32,
        channels=3,
        dim_mults=(1, 2),
        dropout=0.1
    ).to(device)
    
    # Create ElucidatedDiffusion model
    diffusion = ElucidatedDiffusion(
        unet,
        image_size=64,
        channels=3,
        num_sample_steps=4  # Small number for testing
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
        samples = diffusion.sample(self_cond, batch_size=batch_size, num_sample_steps=2, clamp=True)
    
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
            "flash_attn": False
        },
        "normalization_params": {
            "type": "physics",
            "input_norm": [1.0, 1.0, 2.5e-4],
            "output_norm": [1.0, 1.0, 2.5e-4]
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
            "sub_x": 1
        },
        "loss_params": {
            "type": "mse",
            "reduction": "mean"
        },
        "train_loader_params": {
            "batch_size": 4,
            "shuffle": True,
            "num_workers": 2,
            "pin_memory": True
        },
        "optimizer_params": {
            "optimizer_type": "adamw",
            "lr": 1.0e-4,
            "weight_decay": 1.0e-5,
            "betas": [0.9, 0.999],
            "use_scheduler": True,
            "scheduler_type": "cosine",
            "T_max": 100,
            "eta_min": 1e-6
        },
        "train_params": {
            "epochs": 100,
            "ckpt_path": "checkpoints/diffusion_model.pt",
        }
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
    x = torch.randn(batch_size, model.channels, model.image_size, model.image_size).to(device)
    
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
        samples_dpmpp = model.sample_using_dpmpp(x, batch_size=batch_size, num_sample_steps=2)
        assert samples_dpmpp.shape == x.shape
        assert torch.all(samples_dpmpp >= -1.0) and torch.all(samples_dpmpp <= 1.0)


def test_diffusion_registry_consistency():
    """Test that the diffusion model registry contains expected models."""
    # Make sure registry is populated 
    from src.diffusion.models.diffusion_factory import _register_all_diffusion_models, DIFFUSION_MODEL_REGISTRY
    
    # Call registration function to ensure all models are registered
    _register_all_diffusion_models()
    
    # Check that at least elucidated diffusion is registered
    assert "elucidated" in DIFFUSION_MODEL_REGISTRY, "ElucidatedDiffusion should be registered"
    
    # Print available models for debugging
    available_models = list(DIFFUSION_MODEL_REGISTRY.keys())
    print(f"Available diffusion models: {available_models}")
    
    # If more models are added in the future, add assertions for them here
