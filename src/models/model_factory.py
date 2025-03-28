"""Factory for creating neural operator models with support for variants."""

import importlib
from typing import Dict, Callable, Any, Optional

# Registry to store model constructors
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(model_type: str, variant: Optional[str] = None):
    """
    Decorator to register a model constructor function.

    Args:
        model_type: Base model type (e.g., 'tfno', 'uno')
        variant: Optional variant name (e.g., 'time_evolution')

    Returns:
        Decorator function
    """

    def decorator(constructor_fn):
        key = f"{model_type}_{variant}" if variant else model_type
        MODEL_REGISTRY[key] = constructor_fn
        return constructor_fn

    return decorator


def create_model(config):
    """
    Create model based on model type and variant specified in config.

    Args:
        config (dict): Configuration dictionary containing model parameters

    Returns:
        Model instance of the requested type and variant
    """
    model_params = config["model_params"]
    model_type = model_params.get("model_type", "tfno").lower()
    model_variant = model_params.get("model_variant", None)

    # Determine the registry key
    key = f"{model_type}_{model_variant}" if model_variant else model_type

    # Check if the model is registered
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key](model_params)
    else:
        available_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model type/variant: {key}. Available models: {available_models}"
        )


# Register the standard TFNO model
@register_model(model_type="tfno")
def create_tfno_model(model_params):
    """Create a standard PhysicsNormalizedTFNO model instance."""
    from ..models.physics_normalized_tfno import PhysicsNormalizedTFNO

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


# Register the time evolution variant of TFNO
@register_model(model_type="tfno", variant="time_evolution")
def create_tfno_time_evolution_model(model_params):
    """Create a PhysicsNormalizedTFNO model with time evolution capabilities."""
    from ..models.physics_normalized_tfno_time_evolution import PhysicsNormalizedTFNO

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
        # Add any additional parameters specific to time evolution variant
        time_steps=model_params.get("time_steps", 1),
    )

    return model


# Template for UNO registration (uncomment and implement when needed)
# @register_model(model_type="uno")
# def create_uno_model(model_params):
#     """Create a standard UNO model."""
#     from ..models.u_no import UNO
#
#     model = UNO(
#         # UNO-specific parameters
#     )
#
#     return model
