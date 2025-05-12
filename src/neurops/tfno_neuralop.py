"""Tensorized Fourier Neural Operator model."""

import torch
import torch.nn as nn
from neuralop.models import TFNO3d

# from neuralop.layers.channel_mlp import LinearChannelMLP
# from neuralop.layers.legacy_spectral_convolution import SpectralConv

from .model_factory import register_model
from ..activations import create_activation


class TFNO_neuralop(nn.Module):
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


@register_model(model_type="tfno-neuralop")
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
        in_channels=model_params.get("in_dim", 6),
        out_channels=model_params.get("out_dim", 3),
        n_layers=model_params.get("n_layers", 4),
        rank=model_params.get("rank", 4000),
        factorization=model_params.get("factorization", "tucker"),
        positional_embedding=None,
        # positional_embedding=model_params.get("positional_embedding", "grid"),
    )

    # # Using Legacy SpectralConv seems to match the original implementation better,
    # # and there are slight gains in performance, accuracy, and convergence speed
    # # from using it, as the spectral block in the legacy version is naturally larger.
    # n_modes = (
    #     model_params.get("modes1", 8),
    #     model_params.get("modes2", 8),
    #     model_params.get("modes3", 8),
    # )
    #
    # convs = nn.ModuleList(
    #     [
    #         SpectralConv(
    #             model_params.get("width", 8),
    #             model_params.get("width", 8),
    #             n_modes,
    #             n_layers=model_params.get("n_layers", 4),
    #             rank=model_params.get("rank", 4000),
    #             fixed_rank_modes=False,
    #             implementation="factorized",
    #             separable=False,
    #             factorization=model_params.get("factorization", "tucker"),
    #             joint_factorization=True,
    #             fno_block_precision="full",
    #         )
    #         for _ in range(model_params.get("n_layers", 4))
    #     ]
    # )

    # # Alternative lifting and projections
    # model.lifting = LinearChannelMLP(
    #     layers=[model_params.get("in_dim", 6), model_params.get("width", 8)]
    # )
    #
    # model.projection = LinearChannelMLP(
    #     layers=[
    #         model_params.get("width", 8),
    #         model_params.get("fc_dim", 128),
    #         model_params.get("out_dim", 3),
    #     ]
    # )

    # model.fno_blocks.convs = convs

    # Wrap the model
    return TFNO_neuralop(model)
