"""Physics-aware normalization that applies channel-wise scaling."""

import torch
import torch.nn as nn

from .normalization_factory import register_normalization


class PhysicsNormalization(nn.Module):
    """
    Physics-aware normalization using pre-defined scaling factors.

    This normalization applies channel-wise scaling based on domain knowledge
    of the typical magnitudes of physical quantities.

    Args:
        input_norm: List of scaling factors for each input channel
        output_norm: List of scaling factors for each output channel
    """

    def __init__(self, input_norm=None, output_norm=None):
        super().__init__()

        if input_norm is not None:
            self.register_buffer("input_norm", torch.tensor(input_norm))
        else:
            self.input_norm = None

        if output_norm is not None:
            self.register_buffer("output_norm", torch.tensor(output_norm))
        else:
            self.output_norm = None

    def normalize(self, x):
        """Apply normalization to input"""
        if self.input_norm is not None:
            return x / self.input_norm.view(1, -1, 1, 1, 1)
        return x

    def denormalize(self, x):
        """Apply denormalization to output"""
        if self.output_norm is not None:
            return x * self.output_norm.view(1, -1, 1, 1, 1)
        return x


@register_normalization("physics")
def create_physics_normalization(norm_params):
    """Create a physics-aware normalization using pre-defined scaling factors."""
    return PhysicsNormalization(
        input_norm=norm_params.get("input_norm"),
        output_norm=norm_params.get("output_norm"),
    )
