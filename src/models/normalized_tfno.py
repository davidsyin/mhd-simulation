"""Tensorized Fourier Neural Operator model."""

import torch
import torch.nn as nn
from neuralop.models import TFNO3d

from .model_factory import register_model
from ..activations import create_activation


class NormalizedTFNO(nn.Module):
    """
    TFNO3d model wrapper.

    This class wraps the neuraloperator library's TFNO3d implementation.

    Parameters
    ----------
    model : nn.Module
        The core TFNO3d model
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Forward through neural operator model
        return self.model(x)


@register_model(model_type="tfno")
def create_tfno_model(model_params):
    """Create a TFNO3d model from the neuraloperator library."""

    # Get activation function using the factory
    activation_config = model_params.get("activation", "gelu")
    if isinstance(activation_config, str):
        non_linearity = create_activation(activation_config)
    else:
        non_linearity = create_activation(activation_config)

    # Setup domain padding for 3D data [pad_z, pad_x, pad_y]
    pad_z = model_params.get("pad_z", 0.0)
    pad_x = model_params.get("pad_x", 0.0)
    pad_y = model_params.get("pad_y", 0.0)
    domain_padding = None
    if pad_z > 0 or pad_x > 0 or pad_y > 0:
        domain_padding = [pad_z, pad_x, pad_y]

    # Create the TFNO3d model
    model = TFNO3d(
        n_modes_height=model_params.get("modes1", 8),
        n_modes_width=model_params.get("modes2", 8),
        n_modes_depth=model_params.get("modes3", 8),
        hidden_channels=model_params.get("width", 8),
        in_channels=model_params.get("in_dim", 3),
        out_channels=model_params.get("out_dim", 3),
        n_layers=model_params.get("n_layers", 4),
        lifting_channels=model_params.get("fc_dim", 128),
        projection_channels=model_params.get("fc_dim", 128),
        non_linearity=non_linearity,
        rank=model_params.get("rank", 0.5),
        factorization=model_params.get("factorization", "tucker"),
        domain_padding=domain_padding,
        domain_padding_mode=model_params.get("domain_padding_mode", "symmetric"),
        positional_embedding="grid",
    )

    # Wrap the model
    return NormalizedTFNO(model)
