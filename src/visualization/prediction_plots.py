"""Functions for visualizing model predictions and results."""

import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


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
        try:
            x, y = next(iter(val_loader))
            x, y = x.to(device), y.to(device)

            has_grid_embeddings = x.shape[1] > y.shape[1]
            data_channels = y.shape[1]  # Number of data channels (3 for u_x, u_y, A)

            print(f"Input shape: {x.shape}, Output shape: {y.shape}")
            print(f"Detected grid embeddings: {has_grid_embeddings}")

            if has_grid_embeddings:
                data_channel_indices = list(
                    range(x.shape[1] - data_channels, x.shape[1])
                )
                print(f"Using data channels from input: {data_channel_indices}")
            else:
                data_channel_indices = list(range(x.shape[1]))

            pred = model(x)

            if normalizer is not None and hasattr(normalizer, "denormalize"):
                if has_grid_embeddings:
                    # For input tensor, we only want to denormalize the data channels, not the grid
                    x_data_only = x[:, data_channel_indices]
                    x_data_denorm = normalizer.denormalize(x_data_only)

                    x_denorm = x.clone()
                    x_denorm[:, data_channel_indices] = x_data_denorm
                else:
                    # If no grid embeddings, denormalize the whole input
                    x_denorm = normalizer.denormalize(x)

                y_denorm = normalizer.denormalize(y)
                pred_denorm = normalizer.denormalize(pred)
            else:
                # If no normalizer is available, use the data as is
                x_denorm = x
                y_denorm = y
                pred_denorm = pred

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
                    if has_grid_embeddings:
                        input_channel_idx = data_channel_indices[c]
                        im = axes[c, 0].imshow(x_np[i, input_channel_idx, 0], cmap=cmap)
                    else:
                        im = axes[c, 0].imshow(x_np[i, c, 0], cmap=cmap)

                    axes[c, 0].set_title(f"IC: {channel_names[c]}")
                    axes[c, 0].axis("off")
                    plt.colorbar(im, ax=axes[c, 0], fraction=0.046, pad=0.04)

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
