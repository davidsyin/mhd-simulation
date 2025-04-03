"""Validation functions for model evaluation."""

import torch
import torch.nn as nn
from tqdm import tqdm


def validate(model, val_loader, criterion, device):
    """Validate model performance

    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def validate_size_mismatch_fix(model, val_loader, criterion, device):
    """Validate model with size mismatch handling

    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # Handle size mismatch
            if pred.shape[2] == 1 and y.shape[2] > 1:
                y_adjusted = y[:, :, -1:, :, :]
                loss = criterion(pred, y_adjusted)
            else:
                loss = criterion(pred, y)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def validate_with_denorm_metrics(model, val_loader, criterion, device):
    """
    Validate model performance and calculate denormalized MSE loss.

    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        tuple: (normalized_loss, denormalized_loss)
    """
    model.eval()
    normalized_mse = 0.0
    denormalized_mse = 0.0

    # Check if dataset has a normalizer
    has_normalizer = (
        hasattr(val_loader.dataset, "normalizer")
        and val_loader.dataset.normalizer is not None
    )
    mse_criterion = nn.MSELoss(reduction="mean")

    # Create a copy of the normalizer to avoid modifying the original
    normalizer = None
    if has_normalizer:
        import copy

        normalizer = copy.deepcopy(val_loader.dataset.normalizer)

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            pred = model(x)

            # Calculate normalized loss (used for training)
            normalized_loss = criterion(pred, y)
            normalized_mse += normalized_loss.item()

            # Calculate denormalized MSE if normalizer exists
            if normalizer is not None:
                # Make sure normalizer is on the right device
                if hasattr(normalizer, "to"):
                    normalizer = normalizer.to(device)

                # Denormalize predictions and ground truth
                pred_denorm = normalizer.denormalize(pred)
                y_denorm = normalizer.denormalize(y)

                # Calculate denormalized MSE
                denorm_loss = mse_criterion(pred_denorm, y_denorm)
                denormalized_mse += denorm_loss.item()
            else:
                # If no normalizer, denormalized is same as normalized
                denormalized_mse = normalized_mse

    # Calculate averages
    avg_normalized_mse = normalized_mse / len(val_loader)
    avg_denormalized_mse = denormalized_mse / len(val_loader)

    return avg_normalized_mse, avg_denormalized_mse
