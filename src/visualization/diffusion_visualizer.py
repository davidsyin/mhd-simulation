import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


def save_diffusion_samples(
    model, val_loader, device, epoch, log_dir, num_samples=4, normalize=True
):
    """
    Generate and save samples from a diffusion model.

    Args:
        model: Diffusion model
        val_loader: Validation data loader
        device: Device to run generation on
        epoch: Current epoch number
        log_dir: Directory to save samples
        num_samples: Number of samples to generate
        normalize: Whether to normalize images for visualization
    """
    model.eval()

    with torch.no_grad():
        # Get a batch from the validation dataset
        l_fidel_samples, h_fidel_samples = next(iter(val_loader))
        l_fidel_samples = l_fidel_samples[:num_samples].to(device)
        h_fidel_samples = h_fidel_samples[:num_samples].to(device)

        # Generate samples
        samples = model.sample(l_fidel_samples)

        # Handle different channel dimensions for visualization
        def prepare_for_vis(tensor):
            # If multi-channel, choose first channel or select appropriate channels
            if tensor.shape[1] > 1:
                return tensor[:, 0:1]  # Take first channel for visualization
            return tensor

        # Create comparison grid: low-fidelity input, high-fidelity target, model prediction
        try:
            comparison = torch.cat(
                [
                    prepare_for_vis(l_fidel_samples),
                    prepare_for_vis(h_fidel_samples),
                    prepare_for_vis(samples),
                ],
                dim=0,
            )
        except:
            # print(l_fidel_samples.shape, h_fidel_samples.shape, samples.shape)
            comparison = torch.cat(
                [
                    prepare_for_vis(l_fidel_samples[:,:,0:1]),
                    prepare_for_vis(h_fidel_samples),
                    prepare_for_vis(samples),
                ],
                dim=0,
            )

        # Normalize for better visualization if requested
        if normalize:
            comparison = (comparison - comparison.min()) / (
                comparison.max() - comparison.min() + 1e-8
            )

        # Create directory for samples
        sample_dir = os.path.join(log_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)

        # Save visualization
        vutils.save_image(
            comparison,
            os.path.join(sample_dir, f"epoch_{epoch}.png"),
            nrow=num_samples,
            normalize=False,
            padding=2,
        )

        return comparison


def save_diffusion_comparison_grid(
    inputs, targets, predictions, save_path, normalize=True
):
    """
    Save a comparison grid showing inputs, targets, and predictions.

    Args:
        inputs: Low-fidelity inputs (tensor)
        targets: High-fidelity targets (tensor)
        predictions: Model predictions (tensor)
        save_path: Path to save the visualization
        normalize: Whether to normalize for visualization
    """
    # Ensure inputs are on CPU and detached
    inputs = inputs.cpu().detach()
    targets = targets.cpu().detach()
    predictions = predictions.cpu().detach()

    # Handle multi-channel data
    def prepare_for_vis(tensor):
        if tensor.shape[1] > 1:
            return tensor[:, 0:1]
        return tensor

    inputs = prepare_for_vis(inputs)
    targets = prepare_for_vis(targets)
    predictions = prepare_for_vis(predictions)

    # Create grid
    n_samples = inputs.shape[0]
    comparison = torch.cat([inputs, targets, predictions], dim=0)

    # Normalize if requested
    if normalize:
        comparison = (comparison - comparison.min()) / (
            comparison.max() - comparison.min() + 1e-8
        )

    # Save visualization
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vutils.save_image(comparison, save_path, nrow=n_samples, normalize=False, padding=2)

    return comparison
