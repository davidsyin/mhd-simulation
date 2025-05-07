import torch
from .unet import UNet
from .elucidated import ElucidatedDiffusion


def create_diffusion_model(config):
    """Create a diffusion model based on configuration

    Args:
        config: Dictionary containing model configuration

    Returns:
        Configured diffusion model
    """
    model_params = config.get("model_params", {})
    model_type = model_params.get("model_type", "elucidated")

    # Create UNet model
    unet = UNet(
        dim=model_params.get("base_dim", 64),
        dim_mults=model_params.get("dim_mults", (1, 2, 4, 8)),
        channels=model_params.get("channels", 3),
        self_condition=model_params.get("self_condition", True),
        learned_variance=model_params.get("learned_variance", False),
        learned_sinusoidal_cond=model_params.get("learned_sinusoidal_cond", False),
        random_fourier_features=model_params.get("random_fourier_features", False),
        flash_attn=model_params.get("flash_attn", False),
        attn_heads=model_params.get("attn_heads", 4),
        attn_dim_head=model_params.get("attn_dim_head", 32),
        dropout=model_params.get("dropout", 0.0),
    )

    # Wrap UNet in ElucidatedDiffusion model if specified
    if model_type == "elucidated":
        model = ElucidatedDiffusion(
            unet,
            image_size=model_params.get("image_size", 128),
            channels=model_params.get("channels", 3),
            num_sample_steps=model_params.get("num_sample_steps", 32),
            sigma_min=model_params.get("sigma_min", 0.002),
            sigma_max=model_params.get("sigma_max", 80),
            sigma_data=model_params.get("sigma_data", 0.5),
            rho=model_params.get("rho", 7),
            P_mean=model_params.get("P_mean", -1.2),
            P_std=model_params.get("P_std", 1.2),
            S_churn=model_params.get("S_churn", 80),
            S_tmin=model_params.get("S_tmin", 0.05),
            S_tmax=model_params.get("S_tmax", 50),
            S_noise=model_params.get("S_noise", 1.003),
        )
        return model

    # Otherwise just return the UNet directly
    return unet
