import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import TFNO3d

from .model_factory import register_model


class PhysicsNormalizedTFNO(nn.Module):
    def __init__(
        self,
        modes1=8,
        modes2=8,
        modes3=8,
        width=8,
        n_layers=4,
        in_dim=3,
        out_dim=3,
        fc_dim=128,
        activation="gelu",
        pad_x=0.1,
        pad_y=0.1,
        pad_z=0.1,
        input_norm=None,
        output_norm=None,
        rank=0.5,
        factorization="cp",
        **kwargs,
    ):
        super().__init__()

        # Convert activation string to function
        if activation == "gelu":
            nonlinearity = F.gelu
        elif activation == "relu":
            nonlinearity = F.relu
        elif activation == "tanh":
            nonlinearity = torch.tanh
        else:
            raise ValueError(f"Activation {activation} not supported")

        # Setup domain padding for 3D data [pad_z, pad_x, pad_y]
        domain_padding = None
        if pad_z > 0 or pad_x > 0 or pad_y > 0:
            domain_padding = [pad_z, pad_x, pad_y]

        # Setup normalization buffers
        if input_norm is not None:
            self.register_buffer("input_norm", torch.tensor(input_norm))
        else:
            self.input_norm = None

        if output_norm is not None:
            self.register_buffer("output_norm", torch.tensor(output_norm))
        else:
            self.output_norm = None

        # Create the TFNO3d model with proper settings
        self.model = TFNO3d(
            n_modes_height=modes1,
            n_modes_width=modes2,
            n_modes_depth=modes3,
            hidden_channels=width,
            in_channels=in_dim,  # Keep as 3 to match data
            out_channels=out_dim,
            n_layers=n_layers,
            lifting_channels=fc_dim,
            projection_channels=fc_dim,
            non_linearity=nonlinearity,
            rank=rank,
            factorization=factorization,
            domain_padding=domain_padding,
            domain_padding_mode="symmetric",
            positional_embedding=None,  # Disable positional embedding
            **kwargs,
        )

    def forward(self, x):
        # Apply physics-aware normalization
        if self.input_norm is not None:
            x = x / self.input_norm.view(1, -1, 1, 1, 1)

        # Forward through TFNO model
        x = self.model(x)

        # Apply physics-aware denormalization
        if self.output_norm is not None:
            x = x * self.output_norm.view(1, -1, 1, 1, 1)

        return x


@register_model(model_type="tfno")
def create_tfno_model(model_params):
    """Create a standard PhysicsNormalizedTFNO model instance."""
    model = PhysicsNormalizedTFNO(
        modes1=model_params.get("modes1", 8),
        modes2=model_params.get("modes2", 8),
        modes3=model_params.get("modes3", 8),
        width=model_params.get("width", 8),
        n_layers=model_params.get("n_layers", 4),
        in_dim=model_params.get("in_dim", 3),
        out_dim=model_params.get("out_dim", 3),
        fc_dim=model_params.get("fc_dim", 128),
        activation=model_params.get("activation", "gelu"),
        pad_x=model_params.get("pad_x", 0.1),
        pad_y=model_params.get("pad_y", 0.1),
        pad_z=model_params.get("pad_z", 0.1),
        input_norm=model_params.get("input_norm", [1.0, 1.0, 1.0]),
        output_norm=model_params.get("output_norm", [1.0, 1.0, 1.0]),
        rank=model_params.get("rank", 0.5),
        factorization=model_params.get("factorization", "cp"),
    )

    return model
