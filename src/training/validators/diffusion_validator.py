"""Validation functions for diffusion models."""

import torch
import torch.nn as nn
from tqdm import tqdm
import time

from ...losses import LpLoss
from ...utils import denormalize_with_channel_mapping


def validate_diffusion(model, val_loader, device, rel_l2_metric=None):
    """Validate diffusion model performance

    Args:
        model: Diffusion model
        val_loader: DataLoader for validation data
        device: Device to run validation on
        rel_l2_metric: Optional relative L2 error metric

    Returns:
        float: Average validation relative L2 loss
    """
    model.eval()
    total_loss = 0
    data_time = 0
    inference_time = 0
    metric_time = 0

    if rel_l2_metric is None:
        rel_l2_metric = LpLoss(size_average=True)

    with torch.no_grad():
        for l_fidel, h_fidel in tqdm(val_loader, desc="Validating"):
            # Transfer data with non_blocking=True for better performance
            data_start = time.time()
            l_fidel = l_fidel.to(device, non_blocking=True)
            h_fidel = h_fidel.to(device, non_blocking=True)
            data_time += time.time() - data_start

            # Sample from diffusion model (prediction stage)
            inference_start = time.time()
            pred = model.sample(l_fidel)
            inference_time += time.time() - inference_start

            metric_start = time.time()
            loss = rel_l2_metric(pred, h_fidel)
            metric_time += time.time() - metric_start

            total_loss += loss.item()

    print(f"Validation Statistics:")
    print(f"- Data transfer: {data_time:.2f}s")
    print(f"- Inference time: {inference_time:.2f}s")
    print(f"- Metric calculation: {metric_time:.2f}s\n")

    torch.cuda.empty_cache()

    return total_loss / len(val_loader)


def validate_with_denorm_metrics_diffusion(model, val_loader, device, use_denorm=True):
    """
    Validate diffusion model performance and calculate denormalized metrics.

    Args:
        model: Neural diffusion model
        val_loader: DataLoader for validation data
        device: Device to run evaluation on
        use_denorm: Whether to use denormalized metrics

    Returns:
        tuple: (normalized_loss, denormalized_rel_l2, denormalized_mse)
    """
    model.eval()

    # Get dataset and config information
    dataset = val_loader.dataset
    config = getattr(dataset, "config", {})
    dataset_params = config.get("dataset_params", {})

    # Get channel indices for denormalization
    channel_indices = dataset_params.get("channel_indices", None)
    if isinstance(channel_indices, int):
        channel_indices = [channel_indices]
    elif channel_indices is None:
        # If no specific channel indices, use sequential indices up to the number of channels
        model_params = config.get("model_params", {})
        num_channels = model_params.get("channels", 3)
        channel_indices = list(range(min(num_channels, 3)))  # Limit to max 3 channels

    # Get normalizers
    normalizer = None
    if use_denorm:
        normalizer = getattr(dataset, "target_normalizer", None)

    # Initialize metrics
    mse_criterion = nn.MSELoss()
    rel_l2 = LpLoss()

    total_norm_loss = 0.0
    total_denorm_rel_l2 = 0.0
    total_denorm_mse = 0.0
    count = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Get model prediction
            pred = model.sample(x)

            # Calculate normalized loss
            norm_loss = mse_criterion(pred, y)
            total_norm_loss += norm_loss.item()

            # Calculate denormalized metrics if requested
            if use_denorm and normalizer is not None:
                try:
                    # Denormalize using our utility function
                    pred_denorm = denormalize_with_channel_mapping(
                        pred, normalizer, channel_indices
                    )
                    y_denorm = denormalize_with_channel_mapping(
                        y, normalizer, channel_indices
                    )

                    # Calculate denormalized metrics
                    denorm_rel_l2 = rel_l2(pred_denorm, y_denorm)
                    denorm_mse = mse_criterion(pred_denorm, y_denorm)

                    total_denorm_rel_l2 += denorm_rel_l2.item()
                    total_denorm_mse += denorm_mse.item()
                except Exception as e:
                    print(f"Warning: Denormalization failed: {e}")
                    # Fall back to normalized metrics
                    total_denorm_rel_l2 += rel_l2(pred, y).item()
                    total_denorm_mse += norm_loss.item()
            else:
                # Use normalized values if denormalization is disabled
                total_denorm_rel_l2 += rel_l2(pred, y).item()
                total_denorm_mse += norm_loss.item()

            count += 1

    # Calculate averages
    avg_norm_loss = total_norm_loss / count
    avg_denorm_rel_l2 = total_denorm_rel_l2 / count
    avg_denorm_mse = total_denorm_mse / count

    return avg_norm_loss, avg_denorm_rel_l2, avg_denorm_mse
