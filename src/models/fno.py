"""Fourier Neural Operator model."""

import torch
import torch.nn as nn
from physicsnemo.models.fno import FNO

from .model_factory import register_model
from ..activations import create_activation


# Register TFNO models with the repository's registry system
@register_model(model_type="fno", variant=None)
def fno_model_default(params):
    """Default FNO model."""
    return FNO(
        in_channels=params.get("in_channels", 6),
        out_channels=params.get("out_channels", 3),
        decoder_layers=params.get("decoder_layers", 1),
        dimension=params.get("dimension", 3),
        latent_channels=params.get("latent_channels", 32),
        num_fno_layers=params.get("num_fno_layers", 4),
        num_fno_modes=params.get("num_fno_modes", 16),
        padding=params.get("padding", 8),
        padding_type=params.get("padding_type", "constant"),
        activation_fn=params.get("activation_fn", "gelu"),
        coord_features=params.get("coord_features", False),
    )


@register_model(model_type="fno", variant="3d")
def fno_model_3d(params):
    """3D FNO model."""
    params["dimension"] = 3
    return fno_model_default(params)
