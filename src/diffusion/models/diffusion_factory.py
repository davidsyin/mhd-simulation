"""Factory for creating diffusion models with automatic registration."""

from typing import Dict, Callable, Any, Optional

# Registry to store diffusion model constructors
DIFFUSION_MODEL_REGISTRY: Dict[str, Callable] = {}


def register_diffusion_model(model_type: str, variant: Optional[str] = None):
    """
    Decorator to register a diffusion model constructor function.

    Args:
        model_type: Base model type (e.g., 'elucidated', 'ddpm')
        variant: Optional variant name (e.g., 'image', 'conditional')

    Returns:
        Decorator function
    """

    def decorator(constructor_fn):
        key = f"{model_type}_{variant}" if variant else model_type
        DIFFUSION_MODEL_REGISTRY[key.lower()] = constructor_fn
        return constructor_fn

    return decorator


def create_diffusion_model(config):
    """
    Create diffusion model based on model type and variant specified in config.

    Args:
        config (dict): Configuration dictionary containing model parameters

    Returns:
        Diffusion model instance of the requested type and variant
    """
    # Make sure all models are registered
    _register_all_diffusion_models()

    model_params = config.get("model_params", {})
    model_type = model_params.get("model_type", "elucidated").lower()
    model_variant = model_params.get("model_variant", None)

    # Determine the registry key
    key = f"{model_type}_{model_variant}" if model_variant else model_type
    key = key.lower()

    # Check if the model is registered
    if key in DIFFUSION_MODEL_REGISTRY:
        return DIFFUSION_MODEL_REGISTRY[key](model_params)
    else:
        available_models = ", ".join(DIFFUSION_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported diffusion model type/variant: {key}. Available models: {available_models}"
        )


def _register_all_diffusion_models():
    """
    Imports all diffusion model modules to ensure all decorated model
    constructor functions are registered with the registry.
    """
    # Import model modules
    from . import elucidated_diffusion

    # Add imports for future diffusion model files here
