"""U-shaped Neural Operator implementation using the neuraloperator library."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.models.uno import UNO

from .model_factory import register_model


@register_model(model_type="uno")
def create_uno_model(model_params):
    """Create a U-shaped Neural Operator model using neuraloperator library.

    Args:
        model_params: Dictionary containing model parameters from config

    Returns:
        Initialized UNO model
    """
    in_channels = model_params.get("in_channels", 3)
    out_channels = model_params.get("out_channels", 3)
    hidden_channels = model_params.get("latent_channels", 32)

    n_layers = model_params.get("num_uno_layers", 4)
    lifting_channels = model_params.get("lifting_channels", 256)
    projection_channels = model_params.get("projection_channels", 256)

    activation_fn_name = model_params.get("activation_fn", "gelu").lower()
    if activation_fn_name == "gelu":
        non_linearity = F.gelu
    elif activation_fn_name == "relu":
        non_linearity = F.relu
    elif activation_fn_name == "silu" or activation_fn_name == "swish":
        non_linearity = F.silu
    else:
        non_linearity = F.gelu  # Default

    # Number of modes for each layer
    uno_n_modes = model_params.get("uno_n_modes", None)
    if uno_n_modes is None:
        # Create default modes configuration
        num_modes = model_params.get("num_uno_modes", 16)
        dimension = model_params.get("dimension", 2)
        uno_n_modes = [[num_modes] * dimension for _ in range(n_layers)]

    # Output channels for each layer
    uno_out_channels = model_params.get("uno_out_channels", None)
    if uno_out_channels is None:
        # Create default channel configuration
        base_channels = hidden_channels
        # Simple symmetric architecture
        if n_layers == 4:
            uno_out_channels = [
                base_channels,
                base_channels * 2,
                base_channels * 2,
                base_channels,
            ]
        elif n_layers == 5:
            uno_out_channels = [
                base_channels,
                base_channels * 2,
                base_channels * 4,
                base_channels * 2,
                base_channels,
            ]
        else:
            # For any other number of layers, create a simple configuration
            uno_out_channels = [base_channels] * n_layers

    # Scaling factors for each layer
    uno_scalings = model_params.get("uno_scalings", None)
    if uno_scalings is None:
        dimension = model_params.get("dimension", 2)
        # For a standard U-shape: downsample then upsample
        if n_layers >= 4:
            mid = n_layers // 2
            uno_scalings = []
            # Downsampling
            for i in range(mid):
                if i == 0:
                    uno_scalings.append([1.0] * dimension)
                else:
                    uno_scalings.append([0.5] * dimension)
            # Upsampling
            for i in range(n_layers - mid):
                if i == n_layers - mid - 1:
                    uno_scalings.append([1.0] * dimension)
                else:
                    uno_scalings.append([2.0] * dimension)
        else:
            # Default to no scaling for simpler architectures
            uno_scalings = [[1.0] * dimension for _ in range(n_layers)]

    horizontal_skips_map = model_params.get("horizontal_skips_map", None)

    coord_features = model_params.get("coord_features", False)
    positional_embedding = "grid" if coord_features else None

    domain_padding = model_params.get("domain_padding", None)
    domain_padding_mode = model_params.get("domain_padding_mode", "one-sided")

    # IMPORTANT:
    # Use skip types that support different input/output dimensions
    fno_skip = model_params.get("fno_skip", "linear")
    horizontal_skip = model_params.get("horizontal_skip", "linear")
    channel_mlp_skip = model_params.get("channel_mlp_skip", "linear")

    model = UNO(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        lifting_channels=lifting_channels,
        projection_channels=projection_channels,
        positional_embedding=positional_embedding,
        n_layers=n_layers,
        uno_out_channels=uno_out_channels,
        uno_n_modes=uno_n_modes,
        uno_scalings=uno_scalings,
        horizontal_skips_map=horizontal_skips_map,
        non_linearity=non_linearity,
        fno_skip=fno_skip,  # Explicitly set to linear
        horizontal_skip=horizontal_skip,  # Explicitly set to linear
        channel_mlp_skip=channel_mlp_skip,  # Explicitly set to linear
        domain_padding=domain_padding,
        domain_padding_mode=domain_padding_mode,
    )

    return model


@register_model(model_type="uno", variant="3d")
def create_uno_3d_model(model_params):
    """Create a 3D variant of the U-shaped Neural Operator.

    Args:
        model_params: Dictionary containing model parameters from config

    Returns:
        Initialized UNO model with 3D configuration
    """
    model_params = {**model_params, "dimension": 3}

    # Default 3D-specific parameters
    if "uno_n_modes" not in model_params:
        n_layers = model_params.get("num_uno_layers", 4)
        num_modes = model_params.get("num_uno_modes", 8)
        model_params["uno_n_modes"] = [
            [num_modes, num_modes, num_modes] for _ in range(n_layers)
        ]

    return create_uno_model(model_params)
