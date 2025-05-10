import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import wraps, partial
from packaging import version
from collections import namedtuple
from einops import rearrange, reduce, repeat

from ...utils.utils import exists, default, once, print_once
from .normalization import RMSNorm

# constants

AttentionConfig = namedtuple(
    "AttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)

# Attention classes


class Attend(nn.Module):
    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        # Initialize flash attention flag
        self.flash = flash
        self.flash_available = False

        # Check if we can enable flash attention
        if flash and version.parse(torch.__version__) >= version.parse("2.0.0"):
            self.flash_available = True
        else:
            if flash:
                print_once(
                    "Flash attention requested but PyTorch version < 2.0.0, falling back to regular attention\n"
                )
            self.flash_available = False

        # determine efficient attention configs for cuda and cpu
        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not self.flash_available:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        device_version = version.parse(
            f"{device_properties.major}.{device_properties.minor}"
        )

        if device_version >= version.parse("8.0"):
            print_once("\nA100/H100 GPU detected (compute capability >= 8.0)\n")
            # Enable all attention mechanisms as fallback options
            self.cuda_config = AttentionConfig(True, True, True)
        else:
            print_once(
                "\nNon-A100/H100 GPU detected, using math or mem efficient attention if input tensor is on cuda\n"
            )
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )

        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Use appropriate config based on device
        config = self.cuda_config if is_cuda else self.cpu_config

        # Use the new torch.nn.attention.sdpa_kernel instead of torch.backends.cuda.sdp_kernel
        backends = []
        if config.enable_flash:
            backends.append(torch.nn.attention.SDPBackend.FLASH_ATTENTION)
        if config.enable_math:
            backends.append(torch.nn.attention.SDPBackend.MATH)
        if config.enable_mem_efficient:
            backends.append(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)

        # Only use the context manager if we have backends
        if backends:
            with torch.nn.attention.sdpa_kernel(backends):
                out = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.dropout if self.training else 0.0
                )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash and self.flash_available:
            return self.flash_attn(q, k, v)

        # Regular attention computation
        scale = default(self.scale, q.shape[-1] ** -0.5)
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv
        )

        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        mk, mv = map(lambda t: repeat(t, "h c n -> b h c n", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
