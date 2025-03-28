"""Factory for creating loss functions with support for custom implementations."""

import torch
import torch.nn as nn
from typing import Dict, Callable, Any, Optional

# Registry to store loss function constructors
LOSS_REGISTRY: Dict[str, Callable] = {}


def register_loss(loss_type: str):
    """
    Decorator to register a loss function constructor.

    Args:
        loss_type: Name of the loss function

    Returns:
        Decorator function
    """

    def decorator(constructor_fn):
        LOSS_REGISTRY[loss_type.lower()] = constructor_fn
        return constructor_fn

    return decorator


def create_loss(config):
    """
    Create loss function based on type specified in config.

    Args:
        config (dict): Configuration dictionary containing loss parameters

    Returns:
        Loss function instance
    """
    # Get loss configuration
    loss_config = config.get("loss_params", {})
    loss_type = loss_config.get("type", "mse").lower()

    # Check if the loss is registered
    if loss_type in LOSS_REGISTRY:
        return LOSS_REGISTRY[loss_type](loss_config)
    else:
        available_losses = ", ".join(LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unsupported loss type: {loss_type}. Available losses: {available_losses}"
        )


# Register standard PyTorch loss functions
@register_loss("mse")
def create_mse_loss(params):
    """Create Mean Squared Error loss."""
    reduction = params.get("reduction", "mean")
    return nn.MSELoss(reduction=reduction)


@register_loss("l1")
def create_l1_loss(params):
    """Create L1 (Mean Absolute Error) loss."""
    reduction = params.get("reduction", "mean")
    return nn.L1Loss(reduction=reduction)


@register_loss("smoothl1")
def create_smooth_l1_loss(params):
    """Create Smooth L1 loss."""
    reduction = params.get("reduction", "mean")
    beta = params.get("beta", 1.0)
    return nn.SmoothL1Loss(reduction=reduction, beta=beta)


@register_loss("crossentropy")
def create_cross_entropy_loss(params):
    """Create Cross Entropy loss."""
    reduction = params.get("reduction", "mean")
    weight = params.get("weight", None)
    if weight is not None:
        weight = torch.tensor(weight)
    return nn.CrossEntropyLoss(weight=weight, reduction=reduction)


# Example of a custom weighted MSE loss
@register_loss("weighted_mse")
def create_weighted_mse_loss(params):
    """Create a custom weighted MSE loss that applies different weights to different components."""
    weights = params.get("weights", [1.0, 1.0, 1.0])  # Weights for different components
    reduction = params.get("reduction", "mean")

    class WeightedMSELoss(nn.Module):
        def __init__(self, weights, reduction="mean"):
            super().__init__()
            self.weights = torch.tensor(weights)
            self.reduction = reduction

        def forward(self, pred, target):
            # Assuming pred and target have shape [batch, channels, *]
            # and weights has length equal to number of channels
            self.weights = self.weights.to(pred.device)

            # Compute MSE for each channel
            squared_diff = (pred - target) ** 2

            # Apply weights along channel dimension
            if squared_diff.dim() > 2:  # Handle spatial dimensions
                # Compute mean over spatial dimensions for each channel
                channel_mse = squared_diff.mean(dim=tuple(range(2, squared_diff.dim())))
                # Apply weights
                weighted_mse = (channel_mse * self.weights.view(1, -1)).mean(dim=1)
            else:
                weighted_mse = (squared_diff * self.weights).mean(dim=1)

            # Apply reduction method
            if self.reduction == "mean":
                return weighted_mse.mean()
            elif self.reduction == "sum":
                return weighted_mse.sum()
            else:  # 'none'
                return weighted_mse

    return WeightedMSELoss(weights, reduction)


# Example of a physics-informed loss function
@register_loss("physics_informed")
def create_physics_informed_loss(params):
    """
    Create a physics-informed loss that combines data fitting with physical constraints.

    This is just an example - implement according to your specific physics equations.
    """
    data_loss_weight = params.get("data_loss_weight", 1.0)
    physics_loss_weight = params.get("physics_loss_weight", 0.1)
    data_loss_type = params.get("data_loss_type", "mse")

    # Create the base data loss
    data_loss = LOSS_REGISTRY[data_loss_type](params)

    class PhysicsInformedLoss(nn.Module):
        def __init__(self, data_loss, data_weight, physics_weight):
            super().__init__()
            self.data_loss = data_loss
            self.data_weight = data_weight
            self.physics_weight = physics_weight

        def forward(self, pred, target, input_data=None):
            # Data fitting loss
            fit_loss = self.data_loss(pred, target)

            # Physics consistency loss (placeholder - implement your equations)
            # This is where you would implement conservation laws, etc.
            # Example: divergence-free constraint for fluid flow
            physics_loss = self.compute_physics_violation(pred, input_data)

            # Combine losses
            total_loss = (
                self.data_weight * fit_loss + self.physics_weight * physics_loss
            )
            return total_loss

        def compute_physics_violation(self, pred, input_data):
            # Placeholder for physics constraints
            # Implement your specific physics equations here
            # For example, computing divergence of velocity field
            # or enforcing conservation of energy

            # Simple placeholder implementation
            return torch.tensor(0.0, device=pred.device)

    return PhysicsInformedLoss(data_loss, data_loss_weight, physics_loss_weight)
