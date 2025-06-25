"""Enhanced visualization functions for diffusion models with detailed plots."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from ..utils import denormalize_with_channel_mapping


def save_enhanced_diffusion_plots(
    model, val_loader, device, epoch, output_dir, num_samples=4, denormalize=True
):
    """
    Generate comprehensive visualizations of diffusion model predictions in a 2x4 grid per channel:

    Row 1 (Initial condition):
    1. Low-fidelity initial condition (input)
    2. True high-fidelity initial condition (ground truth)
    3. Predicted high-fidelity initial condition (output)
    4. Error between prediction and ground truth

    Row 2 (Final condition):
    1. Low-fidelity final condition (input)
    2. True high-fidelity final condition (ground truth)
    3. Predicted high-fidelity final condition (output)
    4. Error between prediction and ground truth

    Args:
        model: Diffusion model
        val_loader: Validation data loader (with batch_size = timesteps * channels)
        device: Device to run model on
        epoch: Current epoch number
        output_dir: Directory to save plots
        num_samples: Number of simulations to visualize
        denormalize: Whether to denormalize for visualization
    """
    model.eval()

    plots_dir = os.path.join(output_dir, "enhanced_diffusion_plots", f"epoch_{epoch}")
    os.makedirs(plots_dir, exist_ok=True)

    dataset = val_loader.dataset
    config = getattr(dataset, "config", {})
    model_params = config.get("model_params", {})
    dataset_params = config.get("dataset_params", {})

    # Get channel indices
    channel_indices = dataset_params.get("channel_indices", None)
    if isinstance(channel_indices, int):
        channel_indices = [channel_indices]
    elif channel_indices is None:
        # If no specific channel indices, use sequential indices up to the number of channels
        num_channels = (
            model.channels
            if hasattr(model, "channels")
            else model_params.get("channels", 3)
        )
        channel_indices = list(range(min(num_channels, 3)))  # Limit to max 3 channels

    print(f"Using channel indices: {channel_indices}")

    # Map channel indices to physical meaning
    all_channel_names = ["u_x", "u_y", "A"]  # All possible channels
    channel_mappings = [all_channel_names[idx] for idx in channel_indices]

    # Access normalizers from dataset
    input_normalizer = getattr(dataset, "input_normalizer", None)
    target_normalizer = getattr(dataset, "target_normalizer", None)

    timesteps = getattr(dataset, "timesteps", None)
    if not timesteps:
        print(
            "Warning: Timesteps information not available in dataset. Using default value of 26."
        )
        timesteps = 26  # Fallback to default if not available

    try:
        with torch.no_grad():
            # Process each simulation
            for sim_idx in range(num_samples):
                try:
                    # Get one batch (which represents one full simulation)
                    batch_iter = iter(val_loader)
                    # Skip to the simulation we want
                    for _ in range(sim_idx):
                        next(batch_iter)
                    inputs, targets = next(batch_iter)
                except StopIteration:
                    print(f"Only {sim_idx} simulations available in dataset")
                    break

                # Get initial and final conditions
                initial_input = inputs[0].to(device)
                final_input = inputs[-1].to(device)

                initial_target = targets[0].to(device)
                final_target = targets[-1].to(device)

                # Add batch dimension if needed for the model
                if len(initial_input.shape) == 3:  # If shape is (c, nx, ny)
                    initial_input = initial_input.unsqueeze(
                        0
                    )  # Add batch dimension to make (1, c, nx, ny)
                    final_input = final_input.unsqueeze(0)

                # Generate predictions for both initial and final conditions
                initial_prediction = model.sample(initial_input)
                final_prediction = model.sample(final_input)

                # Denormalize if needed
                if (
                    denormalize
                    and input_normalizer is not None
                    and target_normalizer is not None
                ):
                    try:
                        # Make sure all tensors have batch dimension
                        if len(initial_target.shape) == 3:  # (c, nx, ny)
                            initial_target = initial_target.unsqueeze(0)
                        if len(final_target.shape) == 3:  # (c, nx, ny)
                            final_target = final_target.unsqueeze(0)

                        initial_input_denorm = denormalize_with_channel_mapping(
                            initial_input, input_normalizer, channel_indices
                        )

                        initial_target_denorm = denormalize_with_channel_mapping(
                            initial_target, target_normalizer, channel_indices
                        )

                        initial_pred_denorm = denormalize_with_channel_mapping(
                            initial_prediction, target_normalizer, channel_indices
                        )

                        final_input_denorm = denormalize_with_channel_mapping(
                            final_input, input_normalizer, channel_indices
                        )
                        final_target_denorm = denormalize_with_channel_mapping(
                            final_target, target_normalizer, channel_indices
                        )
                        final_pred_denorm = denormalize_with_channel_mapping(
                            final_prediction, target_normalizer, channel_indices
                        )

                    except Exception as e:
                        print(f"Warning: Could not denormalize: {e}")
                        print("Using normalized data instead.")

                        initial_input_denorm = initial_input
                        initial_target_denorm = (
                            initial_target.unsqueeze(0)
                            if len(initial_target.shape) == 3
                            else initial_target
                        )
                        initial_pred_denorm = initial_prediction

                        final_input_denorm = final_input
                        final_target_denorm = (
                            final_target.unsqueeze(0)
                            if len(final_target.shape) == 3
                            else final_target
                        )
                        final_pred_denorm = final_prediction
                else:
                    # If normalization is enabled or normalizers aren't available, use data as is

                    initial_input_denorm = initial_input
                    initial_target_denorm = (
                        initial_target.unsqueeze(0)
                        if len(initial_target.shape) == 3
                        else initial_target
                    )
                    initial_pred_denorm = initial_prediction

                    final_input_denorm = final_input
                    final_target_denorm = (
                        final_target.unsqueeze(0)
                        if len(final_target.shape) == 3
                        else final_target
                    )
                    final_pred_denorm = final_prediction

                initial_input_np = initial_input_denorm.cpu().numpy()
                initial_target_np = initial_target_denorm.cpu().numpy()
                initial_pred_np = initial_pred_denorm.cpu().numpy()

                final_input_np = final_input_denorm.cpu().numpy()
                final_target_np = final_target_denorm.cpu().numpy()
                final_pred_np = final_pred_denorm.cpu().numpy()

                # Number of channels in the data
                n_channels = initial_input_np.shape[1]

                # Process each channel separately
                for c in range(n_channels):
                    ch_name = (
                        channel_mappings[c]
                        if c < len(channel_mappings)
                        else f"Channel {c}"
                    )

                    # Create a 2x4 grid for this channel
                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

                    cmap = "RdBu_r"

                    # Calculate colorbar limits for initial condition based on high-fidelity ground truth
                    vmin_initial = np.min(initial_target_np[0, c])
                    vmax_initial = np.max(initial_target_np[0, c])

                    # Calculate colorbar limits for final condition based on high-fidelity ground truth
                    vmin_final = np.min(final_target_np[0, c])
                    vmax_final = np.max(final_target_np[0, c])

                    # ROW 1: Initial condition processing

                    # Column 1: Low-fidelity initial condition (model input) - using same limits as HF initial
                    im = axes[0, 0].imshow(
                        initial_input_np[0, c],
                        cmap=cmap,
                        vmin=vmin_initial,
                        vmax=vmax_initial,
                    )
                    axes[0, 0].set_title(f"LF Initial (Input): {ch_name}")
                    axes[0, 0].axis("off")
                    cbar = plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

                    # Scientific notation for vector potential
                    if ch_name == "A":
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()

                    # Column 2: True high-fidelity initial condition (ground truth)
                    im = axes[0, 1].imshow(
                        initial_target_np[0, c],
                        cmap=cmap,
                        vmin=vmin_initial,
                        vmax=vmax_initial,
                    )
                    axes[0, 1].set_title(f"HF Initial (True): {ch_name}")
                    axes[0, 1].axis("off")
                    cbar = plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

                    if ch_name == "A":
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()

                    # Column 3: Predicted high-fidelity initial condition (model output)
                    im = axes[0, 2].imshow(
                        initial_pred_np[0, c],
                        cmap=cmap,
                        vmin=vmin_initial,
                        vmax=vmax_initial,
                    )
                    axes[0, 2].set_title(f"HF Initial (Pred): {ch_name}")
                    axes[0, 2].axis("off")
                    cbar = plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

                    if ch_name == "A":
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()

                    # Column 4: Error for initial condition
                    initial_error = initial_pred_np[0, c] - initial_target_np[0, c]

                    # Use actual min and max values for error colormap
                    err_vmin_initial = np.min(initial_error)
                    err_vmax_initial = np.max(initial_error)

                    im = axes[0, 3].imshow(
                        initial_error,
                        cmap="viridis",
                        vmin=err_vmin_initial,
                        vmax=err_vmax_initial,
                    )
                    axes[0, 3].set_title(f"Initial Error: {ch_name}")
                    axes[0, 3].axis("off")
                    cbar = plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)

                    if ch_name == "A":
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()

                    # ROW 2: Final condition processing

                    # Column 1: Low-fidelity final condition (model input) - using same limits as HF final
                    im = axes[1, 0].imshow(
                        final_input_np[0, c],
                        cmap=cmap,
                        vmin=vmin_final,
                        vmax=vmax_final,
                    )
                    axes[1, 0].set_title(f"LF Final (Input): {ch_name}")
                    axes[1, 0].axis("off")
                    cbar = plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

                    if ch_name == "A":
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()

                    # Column 2: True high-fidelity final condition (ground truth)
                    im = axes[1, 1].imshow(
                        final_target_np[0, c],
                        cmap=cmap,
                        vmin=vmin_final,
                        vmax=vmax_final,
                    )
                    axes[1, 1].set_title(f"HF Final (True): {ch_name}")
                    axes[1, 1].axis("off")
                    cbar = plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

                    if ch_name == "A":
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()

                    # Column 3: Predicted high-fidelity final condition (model output)
                    im = axes[1, 2].imshow(
                        final_pred_np[0, c], cmap=cmap, vmin=vmin_final, vmax=vmax_final
                    )
                    axes[1, 2].set_title(f"HF Final (Pred): {ch_name}")
                    axes[1, 2].axis("off")
                    cbar = plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

                    if ch_name == "A":
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()

                    # Column 4: Error for final condition
                    final_error = final_pred_np[0, c] - final_target_np[0, c]

                    # Use actual min and max values for error colormap
                    err_vmin_final = np.min(final_error)
                    err_vmax_final = np.max(final_error)

                    im = axes[1, 3].imshow(
                        final_error,
                        cmap="viridis",
                        vmin=err_vmin_final,
                        vmax=err_vmax_final,
                    )
                    axes[1, 3].set_title(f"Final Error: {ch_name}")
                    axes[1, 3].axis("off")
                    cbar = plt.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

                    if ch_name == "A":
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()

                    plt.tight_layout(rect=[0, 0, 1, 0.97])

                    # Save the figure - separate plot for each channel
                    plt.savefig(
                        os.path.join(
                            plots_dir, f"simulation_{sim_idx}_channel_{c}.png"
                        ),
                        dpi=150,
                    )
                    plt.close(fig)

            print(f"Enhanced diffusion plots for epoch {epoch} saved to {plots_dir}")
            return plots_dir

    except Exception as e:
        print(
            f"Warning: Could not create enhanced diffusion plots for epoch {epoch}: {e}"
        )
        import traceback

        traceback.print_exc()
        print("Continuing with training...")
        return None
