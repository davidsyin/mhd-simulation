"""Validation functions for model evaluation."""

import torch
import torch.nn as nn
from tqdm import tqdm
import time


def validate(model, val_loader, criterion, device):
    """Validate model performance with optimized memory handling

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
    data_time = 0
    forward_time = 0

    # Check if we're using physics-informed loss
    use_physics_loss = (
        hasattr(criterion, "requires_input_data") and criterion.requires_input_data
    )

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            # Transfer data with non_blocking=True for better performance
            data_start = time.time()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            data_time += time.time() - data_start

            forward_start = time.time()
            x = x.contiguous()  # Ensure input is contiguous
            pred = model(x)
            if use_physics_loss:
                loss = criterion(pred, y, x)
            else:
                loss = criterion(pred, y)
            forward_time += time.time() - forward_start

            total_loss += loss.item()

    print(f"Validation Statistics:")
    print(f"- Data transfer: {data_time:.2f}s")
    print(f"- Forward pass: {forward_time:.2f}s")

    # Free memory
    torch.cuda.empty_cache()

    return total_loss / len(val_loader)


def validate_with_denorm_metrics(model, val_loader, criterion, device):
    """
    Validate model performance and calculate denormalized MSE loss.
    Optimized for better performance with timing measurements.

    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        tuple: Avg(total_model_loss, total_denorm_mse)
    """
    model.eval()
    total_model_loss = 0.0
    total_denorm_mse = 0.0
    data_time = 0
    forward_time = 0
    denorm_time = 0

    has_normalizer = (
        hasattr(val_loader.dataset, "normalizer")
        and val_loader.dataset.normalizer is not None
    )
    mse_criterion = nn.MSELoss(reduction="mean")

    # Check if we're using physics-informed loss
    use_physics_loss = (
        hasattr(criterion, "requires_input_data") and criterion.requires_input_data
    )

    # Create a copy of the normalizer to avoid modifying the original
    normalizer = None
    if has_normalizer:
        import copy

        normalizer = copy.deepcopy(val_loader.dataset.normalizer)

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            # Transfer data with non_blocking=True for better performance
            data_start = time.time()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            data_time += time.time() - data_start

            forward_start = time.time()
            x = x.contiguous()  # Ensure input is contiguous
            pred = model(x)
            forward_time += time.time() - forward_start

            # Calculate normalized loss (used for training)
            if use_physics_loss:
                model_loss = criterion(val_loader, pred, y, x)
                total_model_loss += model_loss.item()
            else:
                model_loss = criterion(pred, y)
                total_model_loss += model_loss.item()

            # Calculate denormalized MSE if normalizer exists
            if normalizer is not None:
                denorm_start = time.time()
                if hasattr(normalizer, "to"):
                    normalizer = normalizer.to(device)

                pred_denorm = normalizer.denormalize(pred)
                y_denorm = normalizer.denormalize(y)

                denorm_loss = mse_criterion(pred_denorm, y_denorm)
                total_denorm_mse += denorm_loss.item()
                denorm_time += time.time() - denorm_start
            else:
                total_denorm_mse = total_model_loss

    avg_total_model_loss = total_model_loss / len(val_loader)
    avg_total_denorm_mse = total_denorm_mse / len(val_loader)

    print(f"Validation Loop Statistics:")
    print(f"- Data transfer: {data_time:.2f}s")
    print(f"- Forward pass: {forward_time:.2f}s")
    if normalizer is not None:
        print(f"- Denormalization: {denorm_time:.2f}s")

    print("=" * 30)
    torch.cuda.empty_cache()

    return avg_total_model_loss, avg_total_denorm_mse
