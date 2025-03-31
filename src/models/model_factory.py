"""Factory for creating neural operator models with automatic registration."""

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
        MODEL_REGISTRY[key.lower()] = constructor_fn
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
    # Make sure all models are registered
    _register_all_models()

    model_params = config["model_params"]
    model_type = model_params.get("model_type", "tfno").lower()
    model_variant = model_params.get("model_variant", None)

    # Determine the registry key
    key = f"{model_type}_{model_variant}" if model_variant else model_type
    key = key.lower()

    # Check if the model is registered
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key](model_params)
    else:
        available_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model type/variant: {key}. Available models: {available_models}"
        )


def _register_all_models():
    """
    Imports all model modules to ensure all decorated model
    constructor functions are registered with the registry.
    """
    # Import all model modules
    from . import normalized_tfno
    from . import time_evolution_tfno

    # Add imports for future model files here
