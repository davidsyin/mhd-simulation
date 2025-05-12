"""Functions for visualizing model predictions and results."""

import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from ..utils import (
    get_dataset_normalizer,
    identify_data_channels,
    apply_denormalization,
)


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

    normalizer = get_dataset_normalizer(val_loader)

    with torch.no_grad():
        try:
            x, y = next(iter(val_loader))
            x, y = x.to(device), y.to(device)

            data_channel_indices, has_grid_embeddings, data_channels = (
                identify_data_channels(x, y)
            )

            pred = model(x)

            x_denorm, y_denorm, pred_denorm = apply_denormalization(
                x, y, pred, normalizer, data_channel_indices, has_grid_embeddings
            )

            x_np = x_denorm.cpu().numpy()
            y_np = y_denorm.cpu().numpy()
            pred_np = pred_denorm.cpu().numpy()

            for i in range(min(num_samples, x.shape[0])):
                fig, axes = plt.subplots(3, 4, figsize=(20, 15))

                # Channel names for better interpretability
                channel_names = ["u_x", "u_y", "A"]
                cmap = "RdBu_r"

                # First column: initial conditions
                for c in range(data_channels):
                    im = axes[c, 0].imshow(x_np[i, c, 0], cmap=cmap)

                    axes[c, 0].set_title(f"IC: {channel_names[c]}")
                    axes[c, 0].axis("off")
                    cbar = plt.colorbar(im, ax=axes[c, 0], fraction=0.046, pad=0.04)

                    # Add scientific notation for the vector potential (A) channel - Initial Condition
                    if c == 2:  # For channel A
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()

                # For each channel, calculate colorbar limits based on true results
                for c in range(data_channels):
                    # Make sure target shape is compatible
                    time_dim = min(y_np.shape[2] - 1, -1)  # Use last time step

                    # Second column: true final state
                    vmin = np.min(y_np[i, c, time_dim])
                    vmax = np.max(y_np[i, c, time_dim])

                    # True final state
                    im = axes[c, 1].imshow(
                        y_np[i, c, time_dim], cmap=cmap, vmin=vmin, vmax=vmax
                    )
                    axes[c, 1].set_title(f"True Final: {channel_names[c]}")
                    axes[c, 1].axis("off")
                    cbar = plt.colorbar(im, ax=axes[c, 1], fraction=0.046, pad=0.04)

                    # Use scientific notation for the third channel (A) - True Final
                    if c == 2:  # For channel A
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()

                    # Third column: predicted final state (using same colorbar range as true results)
                    pred_time_dim = min(pred_np.shape[2] - 1, time_dim)
                    im = axes[c, 2].imshow(
                        pred_np[i, c, pred_time_dim], cmap=cmap, vmin=vmin, vmax=vmax
                    )
                    axes[c, 2].set_title(f"Pred Final: {channel_names[c]}")
                    axes[c, 2].axis("off")

                    # Use scientific notation for the third channel (A)
                    cbar = plt.colorbar(im, ax=axes[c, 2], fraction=0.046, pad=0.04)
                    if c == 2:  # For channel A
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()

                    # Fourth column: prediction error (pred - true)
                    error = pred_np[i, c, pred_time_dim] - y_np[i, c, time_dim]

                    # Use the actual min and max values for error colormap
                    err_vmin = np.min(error)
                    err_vmax = np.max(error)

                    im = axes[c, 3].imshow(
                        error, cmap="viridis", vmin=err_vmin, vmax=err_vmax
                    )
                    axes[c, 3].set_title(f"Error (pred - true): {channel_names[c]}")
                    axes[c, 3].axis("off")

                    cbar = plt.colorbar(im, ax=axes[c, 3], fraction=0.046, pad=0.04)
                    # Apply scientific notation for the third channel (A)
                    if c == 2:  # For channel A
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()

                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"sample_{i}.png"))
                plt.close(fig)
                print(
                    f"Comparison plot for epoch {epoch} has successfully been saved under {plots_dir}"
                )

        except RuntimeError as e:
            # Handle device mismatch errors by logging and continuing
            print(f"Warning: Could not create plots for epoch {epoch}: {e}")
            import traceback

            traceback.print_exc()  # Print full traceback for debugging
            print("Continuing with training...")
            return
