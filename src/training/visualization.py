import os
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

    with torch.no_grad():
        # Get a batch of validation data
        x, y = next(iter(val_loader))
        x, y = x.to(device), y.to(device)

        # Forward pass
        pred = model(x)

        # Convert to numpy for plotting
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()

        # Plot predictions
        for i in range(min(num_samples, x.shape[0])):
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))

            # Channel names for better interpretability
            channel_names = ["u_x", "u_y", "A"]

            # First row: initial conditions
            for c in range(3):
                axes[0, c].imshow(x_np[i, c, 0])
                axes[0, c].set_title(f"IC: {channel_names[c]}")
                axes[0, c].axis("off")

            # Second row: true final state
            for c in range(3):
                axes[1, c].imshow(y_np[i, c, -1])
                axes[1, c].set_title(f"True Final: {channel_names[c]}")
                axes[1, c].axis("off")

            # Third row: predicted final state
            for c in range(3):
                axes[2, c].imshow(pred_np[i, c, -1])
                axes[2, c].set_title(f"Pred Final: {channel_names[c]}")
                axes[2, c].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"sample_{i}.png"))
            plt.close(fig)
