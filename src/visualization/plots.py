"""Functions for visualizing model predictions and results."""

import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_prediction_plots(model, val_loader, device, epoch, output_dir, num_samples=2):
    """Save prediction plots to disk

    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        device: Device to run model on
        epoch: Current epoch
        output_dir: Directory to save plots
        num_samples: Number of samples to plot
    """
    model.eval()

    plots_dir = os.path.join(output_dir, "prediction_plots", f"epoch_{epoch}")
    os.makedirs(plots_dir, exist_ok=True)

    # Check if the dataset has a normalizer attribute
    has_dataset_normalizer = (
        hasattr(val_loader.dataset, "normalizer")
        and val_loader.dataset.normalizer is not None
    )

    if has_dataset_normalizer:
        # Save original normalizer device to restore it later
        original_normalizer_device = None
        normalizer = val_loader.dataset.normalizer

        # Record the original device of normalizer buffers
        for name, buffer in normalizer.named_buffers():
            original_normalizer_device = buffer.device
            break

        # Create a GPU copy rather than moving the original
        if hasattr(normalizer, "to"):
            normalizer = copy.deepcopy(normalizer).to(device)
            print(
                f"Using dataset-level {normalizer.__class__.__name__} to denormalize for visualization"
            )
    else:
        normalizer = None
        print("No normalization found - visualizing raw data")

    with torch.no_grad():
        # Get a batch of validation data
        try:
            x, y = next(iter(val_loader))
            x, y = x.to(device), y.to(device)

            # Forward pass
            pred = model(x)

            # Denormalize the data if a normalizer is available
            if normalizer is not None and hasattr(normalizer, "denormalize"):
                # Denormalize predictions and ground truth
                x_denorm = normalizer.denormalize(x)
                y_denorm = normalizer.denormalize(y)
                pred_denorm = normalizer.denormalize(pred)
            else:
                # If no normalizer is available, use the data as is
                x_denorm = x
                y_denorm = y
                pred_denorm = pred

            # Convert to numpy for plotting (must be on CPU)
            x_np = x_denorm.cpu().numpy()
            y_np = y_denorm.cpu().numpy()
            pred_np = pred_denorm.cpu().numpy()

            # Plot predictions
            for i in range(min(num_samples, x.shape[0])):
                # Change to 3x4 subplot layout
                fig, axes = plt.subplots(3, 4, figsize=(20, 15))

                # Channel names for better interpretability
                channel_names = ["u_x", "u_y", "A"]
                cmap = "RdBu_r"  # Use RdBu_r colormap for all plots

                # First column: initial conditions
                for c in range(3):
                    im = axes[c, 0].imshow(x_np[i, c, 0], cmap=cmap)
                    axes[c, 0].set_title(f"IC: {channel_names[c]}")
                    axes[c, 0].axis("off")
                    plt.colorbar(im, ax=axes[c, 0], fraction=0.046, pad=0.04)

                # For each channel, calculate colorbar limits based on true results
                for c in range(3):
                    # Second column: true final state
                    vmin = np.min(y_np[i, c, -1])
                    vmax = np.max(y_np[i, c, -1])

                    # Ensure symmetric colorbar limits for diverging colormap
                    abs_max = max(abs(vmin), abs(vmax))
                    vmin, vmax = -abs_max, abs_max

                    # True final state
                    im = axes[c, 1].imshow(
                        y_np[i, c, -1], cmap=cmap, vmin=vmin, vmax=vmax
                    )
                    axes[c, 1].set_title(f"True Final: {channel_names[c]}")
                    axes[c, 1].axis("off")
                    plt.colorbar(im, ax=axes[c, 1], fraction=0.046, pad=0.04)

                    # Third column: predicted final state (using same colorbar range as true results)
                    im = axes[c, 2].imshow(
                        pred_np[i, c, -1], cmap=cmap, vmin=vmin, vmax=vmax
                    )
                    axes[c, 2].set_title(f"Pred Final: {channel_names[c]}")
                    axes[c, 2].axis("off")
                    plt.colorbar(im, ax=axes[c, 2], fraction=0.046, pad=0.04)

                    # Fourth column: absolute error
                    error = np.abs(y_np[i, c, -1] - pred_np[i, c, -1])
                    im = axes[c, 3].imshow(
                        error, cmap="viridis"
                    )  # Use viridis for error (non-diverging data)
                    axes[c, 3].set_title(f"Abs Error: {channel_names[c]}")
                    axes[c, 3].axis("off")
                    plt.colorbar(im, ax=axes[c, 3], fraction=0.046, pad=0.04)

                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"sample_{i}.png"))
                plt.close(fig)
                print(
                    f"Comparison plot for epoch {epoch} has successfully been saved under {plots_dir}"
                )

        except RuntimeError as e:
            # Handle device mismatch errors by logging and continuing
            print(f"Warning: Could not create plots for epoch {epoch}: {e}")
            print("Continuing with training...")
            return
