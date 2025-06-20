import torch
from tqdm import tqdm
import time
import numpy as np
from ...losses import LpLoss


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

    # Use provided rel_l2 metric or create one
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

            # Calculate error metric
            metric_start = time.time()
            loss = rel_l2_metric(pred, h_fidel)
            metric_time += time.time() - metric_start

            total_loss += loss.item()

    # Log performance statistics
    print(f"Validation Statistics:")
    print(f"- Data transfer: {data_time:.2f}s")
    print(f"- Inference time: {inference_time:.2f}s")
    print(f"- Metric calculation: {metric_time:.2f}s\n")

    # Free memory
    torch.cuda.empty_cache()

    return total_loss / len(val_loader)


def validate_with_denorm_metrics_diffusion(model, val_loader, device, use_denorm=False):
    """
    Validate diffusion model and calculate metrics.

    Args:
        model: Diffusion model
        val_loader: DataLoader for validation data
        device: Device to run validation on
        use_denorm: Whether to use denormalized metrics (default: True)

    Returns:
        tuple: (val_loss, rel_l2, mse) where rel_l2 and mse are either normalized or denormalized
               based on use_denorm parameter
    """
    model.eval()
    val_loss = 0.0
    total_rel_l2 = 0.0
    total_mse = 0.0

    rel_l2_criterion = LpLoss(size_average=True)
    mse_criterion = torch.nn.MSELoss()

    # Get the appropriate normalizer
    normalizer = None
    if use_denorm:
        if (
            hasattr(val_loader.dataset, "target_normalizer")
            and val_loader.dataset.target_normalizer is not None
        ):
            import copy

            normalizer = copy.deepcopy(val_loader.dataset.target_normalizer)

    print(f"Using {'denormalized' if normalizer is not None else 'normalized'} metrics")

    with torch.no_grad():
        for l_fidel, h_fidel in tqdm(val_loader, desc="Validating"):
            l_fidel = l_fidel.to(device, non_blocking=True)
            h_fidel = h_fidel.to(device, non_blocking=True)

            # Sample from diffusion model
            pred = model.sample(l_fidel)

            # Calculate normalized loss
            norm_loss = rel_l2_criterion(pred, h_fidel)
            val_loss += norm_loss.item()

            # Calculate additional metrics - either denormalized or normalized
            if normalizer is not None:
                if hasattr(normalizer, "to"):
                    normalizer = normalizer.to(device)

                pred_denorm = normalizer.denormalize(pred)
                h_fidel_denorm = normalizer.denormalize(h_fidel)

                rel_l2_val = rel_l2_criterion(pred_denorm, h_fidel_denorm)
                mse_val = mse_criterion(pred_denorm, h_fidel_denorm)
            else:
                # Use normalized values if denorm is disabled or no normalizer exists
                rel_l2_val = norm_loss  # Already calculated
                mse_val = mse_criterion(pred, h_fidel)

            total_rel_l2 += rel_l2_val.item()
            total_mse += mse_val.item()

    # Average losses
    avg_val_loss = val_loss / len(val_loader)
    avg_rel_l2 = total_rel_l2 / len(val_loader)
    avg_mse = total_mse / len(val_loader)

    # Free memory
    torch.cuda.empty_cache()

    return avg_val_loss, avg_rel_l2, avg_mse
