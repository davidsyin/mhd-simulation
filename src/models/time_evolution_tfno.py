"""Time evolution variant of the TFNO model for dynamic system simulation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import TFNO3d

from .model_factory import register_model
from ..normalizations import create_normalization
from ..activations import create_activation


class TimeEvolutionTFNO(nn.Module):
    """
    TFNO model specialized for time evolution tasks.

    This model takes an initial condition and evolves it through time,
    producing a time series of outputs using the TFNO architecture.

    Parameters
    ----------
    model : nn.Module
        The core TFNO model
    normalization : nn.Module
        Normalization module to apply to inputs and outputs
    time_steps : int
        Number of time steps to evolve the system
    """

    def __init__(self, model, normalization=None, time_steps=1):
        super().__init__()
        self.model = model
        self.normalization = normalization
        self.time_steps = time_steps

    def forward(self, x):
        # Apply normalization to input if defined
        if self.normalization is not None:
            x = self.normalization.normalize(x)

        # List to store outputs at each time step
        outputs = []
        current = x

        # Evolve the system through time
        for _ in range(self.time_steps):
            # Forward through neural operator model
            current = self.model(current)
            outputs.append(current)

        # Stack outputs along a new time dimension
        x = torch.stack(outputs, dim=2)

        # Apply denormalization to output if defined
        if self.normalization is not None:
            x = self.normalization.denormalize(x)

        return x


@register_model(model_type="tfno", variant="time_evolution")
def create_tfno_time_evolution_model(model_params):
    """Create a TFNO model with time evolution capabilities."""

    # Get activation function using the factory
    activation_config = model_params.get("activation", "gelu")
    if isinstance(activation_config, str):
        # For backward compatibility with string-based activation specification
        non_linearity = create_activation(activation_config)
    else:
        # For new config-based activation specification
        non_linearity = create_activation(activation_config)

    # Setup domain padding for 3D data [pad_z, pad_x, pad_y]
    pad_z = model_params.get("pad_z", 0.0)
    pad_x = model_params.get("pad_x", 0.0)
    pad_y = model_params.get("pad_y", 0.0)
    domain_padding = None
    if pad_z > 0 or pad_x > 0 or pad_y > 0:
        domain_padding = [pad_z, pad_x, pad_y]

    # Create the TFNO3d model for time evolution
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

    # Create normalization if specified
    if model_params.get("use_normalization", True) and (
        model_params.get("input_norm") is not None
        or model_params.get("output_norm") is not None
    ):
        # Create a physics normalization
        normalization_params = {
            "type": "physics",
            "input_norm": model_params.get("input_norm"),
            "output_norm": model_params.get("output_norm"),
        }

        normalization = create_normalization(
            {"normalization_params": normalization_params}
        )
    else:
        normalization = None

    # Wrap the model with time evolution capabilities
    return TimeEvolutionTFNO(
        model, normalization, time_steps=model_params.get("time_steps", 1)
    )
