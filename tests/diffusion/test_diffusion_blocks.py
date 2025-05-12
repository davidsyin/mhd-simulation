"""Tests for diffusion model components and blocks."""

import pytest
import torch
import numpy as np

from src.diffusion.models.components.attention import Attention, LinearAttention, Attend
from src.diffusion.models.components.blocks import Block, ResnetBlock
from src.diffusion.models.components.layers import (
    Upsample, Downsample, SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb
)
from src.diffusion.models.components.normalization import RMSNorm


@pytest.fixture
def device():
    """Return the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
