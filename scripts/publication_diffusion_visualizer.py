"""
Publication-quality visualization functions for diffusion models with 4-timestep plots.
For the easiest usage, add this under src/visualization and run the checkpointed training for a single epoch.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def save_publication_diffusion_plots(
    model,
    val_loader,
    device,
    output_dir,
    split_name="val",
    num_samples=4,
    denormalize=True,
):
    """
    Generate publication-quality visualizations of diffusion model predictions across 4 timesteps.

    This is a direct extension of the enhanced_diffusion_visualizer.py code to support:
    - 4 timesteps instead of just initial and final
    - Publication-quality formatting with (a), (b), (c), (d) labels
    - Saving in high-resolution PNG and PDF formats

    Args:
        model: Diffusion model
        val_loader: Validation data loader (with batch_size = timesteps * channels)
        device: Device to run model on
        output_dir: Directory to save plots
        split_name: Name of the data split (val, test)
        num_samples: Number of simulations to visualize
        denormalize: Whether to denormalize for visualization
    """
    model.eval()

    plots_dir = os.path.join(output_dir, f"{split_name}_publication_plots")
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
            for sim_idx in range(num_samples):
                try:
                    batch_iter = iter(val_loader)
                    for _ in range(sim_idx):
                        try:
                            next(batch_iter)
                        except StopIteration:
                            print(f"Only {sim_idx} simulations available in dataset")
                            break
                    inputs, targets = next(batch_iter)
                except StopIteration:
                    print(f"Only {sim_idx} simulations available in dataset")
                    break

                # Select 4 evenly spaced timesteps (0%, 33%, 67%, 100%)
                if timesteps and timesteps > 4:
                    timestep_indices = [
                        0,
                        timesteps // 3,
                        2 * timesteps // 3,
                        timesteps - 1,
                    ]
                else:
                    # Use whatever we have
                    timestep_indices = list(range(min(4, len(inputs))))
                    print(
                        f"Warning: Not enough timesteps. Using indices: {timestep_indices}"
                    )

                # Get inputs and targets for these timesteps
                timestep_inputs = []
                timestep_targets = []
                for idx in timestep_indices:
                    timestep_inputs.append(inputs[idx].to(device))
                    timestep_targets.append(targets[idx].to(device))

                # Add batch dimension if needed for the model
                for i in range(len(timestep_inputs)):
                    if len(timestep_inputs[i].shape) == 3:  # If shape is (c, nx, ny)
                        timestep_inputs[i] = timestep_inputs[i].unsqueeze(0)
                    if len(timestep_targets[i].shape) == 3:
                        timestep_targets[i] = timestep_targets[i].unsqueeze(0)

                # Generate predictions for each timestep
                timestep_predictions = []
                for inp in timestep_inputs:
                    pred = model.sample(inp)
                    timestep_predictions.append(pred)

                # Denormalize if needed
                if (
                    denormalize
                    and input_normalizer is not None
                    and target_normalizer is not None
                ):
                    try:

                        def denormalize_with_channel_mapping(
                            tensor, normalizer, indices
                        ):
                            bs, c, *spatial_dims = tensor.shape

                            tensor_5d = tensor.unsqueeze(2)  # (bs, c, 1, nx, ny)

                            full_tensor = torch.zeros(
                                bs, 3, 1, *spatial_dims, device=tensor.device
                            )

                            for i, channel_idx in enumerate(indices):
                                if i < c:
                                    full_tensor[:, channel_idx] = tensor_5d[:, i]

                            denormalized = normalizer.denormalize(full_tensor)

                            result = torch.zeros(
                                bs, c, *spatial_dims, device=tensor.device
                            )
                            for i, channel_idx in enumerate(indices):
                                if i < c:
                                    result[:, i] = denormalized[:, channel_idx, 0]

                            return result

                        timestep_inputs_denorm = []
                        timestep_targets_denorm = []
                        timestep_preds_denorm = []

                        for i in range(len(timestep_inputs)):
                            timestep_inputs_denorm.append(
                                denormalize_with_channel_mapping(
                                    timestep_inputs[i],
                                    input_normalizer,
                                    channel_indices,
                                )
                            )

                            timestep_targets_denorm.append(
                                denormalize_with_channel_mapping(
                                    timestep_targets[i],
                                    target_normalizer,
                                    channel_indices,
                                )
                            )

                            timestep_preds_denorm.append(
                                denormalize_with_channel_mapping(
                                    timestep_predictions[i],
                                    target_normalizer,
                                    channel_indices,
                                )
                            )

                    except Exception as e:
                        print(f"Warning: Could not denormalize: {e}")
                        print("Using normalized data instead.")
                        timestep_inputs_denorm = timestep_inputs
                        timestep_targets_denorm = timestep_targets
                        timestep_preds_denorm = timestep_predictions
                else:
                    timestep_inputs_denorm = timestep_inputs
                    timestep_targets_denorm = timestep_targets
                    timestep_preds_denorm = timestep_predictions

                timestep_inputs_np = [t.cpu().numpy() for t in timestep_inputs_denorm]
                timestep_targets_np = [t.cpu().numpy() for t in timestep_targets_denorm]
                timestep_preds_np = [t.cpu().numpy() for t in timestep_preds_denorm]

                n_channels = timestep_inputs_np[0].shape[1]

                for c in range(n_channels):
                    ch_name = (
                        channel_mappings[c]
                        if c < len(channel_mappings)
                        else f"Channel {c}"
                    )

                    # Create a 4×4 grid for publication plots (4 timesteps × 4 columns)
                    fig, axes = plt.subplots(4, 4, figsize=(20, 20))

                    cmap = "RdBu_r"

                    # Process each timestep (rows)
                    for t in range(len(timestep_indices)):
                        # Calculate colorbar limits for target and prediction based on high-fidelity ground truth
                        target_vmin = np.min(timestep_targets_np[t][0, c])
                        target_vmax = np.max(timestep_targets_np[t][0, c])

                        # Calculate separate colorbar limits for input based on its own values
                        input_vmin = np.min(timestep_inputs_np[t][0, c])
                        input_vmax = np.max(timestep_inputs_np[t][0, c])

                        # Column 1: Low-fidelity input - with its own colorbar range
                        im = axes[t, 0].imshow(
                            timestep_inputs_np[t][0, c],
                            cmap=cmap,
                            vmin=input_vmin,  # Use input's own value range
                            vmax=input_vmax,  # Use input's own value range
                        )
                        axes[t, 0].axis("off")
                        cbar = fig.colorbar(im, ax=axes[t, 0], fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=14)

                        # Scientific notation for vector potential
                        if ch_name == "A":
                            formatter = ScalarFormatter(useMathText=True)
                            formatter.set_scientific(True)
                            formatter.set_powerlimits((-2, 2))
                            cbar.formatter = formatter
                            cbar.update_ticks()

                            cbar.ax.yaxis.get_offset_text().set_fontsize(14)

                        # Column 2: True high-fidelity condition (ground truth) - with target range
                        im = axes[t, 1].imshow(
                            timestep_targets_np[t][0, c],
                            cmap=cmap,
                            vmin=target_vmin,  # Use target's range
                            vmax=target_vmax,  # Use target's range
                        )
                        axes[t, 1].axis("off")
                        cbar = fig.colorbar(im, ax=axes[t, 1], fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=14)

                        if ch_name == "A":
                            formatter = ScalarFormatter(useMathText=True)
                            formatter.set_scientific(True)
                            formatter.set_powerlimits((-2, 2))
                            cbar.formatter = formatter
                            cbar.update_ticks()

                            cbar.ax.yaxis.get_offset_text().set_fontsize(14)

                        # Column 3: Predicted high-fidelity condition (model output) - with target range
                        im = axes[t, 2].imshow(
                            timestep_preds_np[t][0, c],
                            cmap=cmap,
                            vmin=target_vmin,  # Use target's range for fair comparison
                            vmax=target_vmax,  # Use target's range for fair comparison
                        )
                        axes[t, 2].axis("off")
                        cbar = fig.colorbar(im, ax=axes[t, 2], fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=14)

                        if ch_name == "A":
                            formatter = ScalarFormatter(useMathText=True)
                            formatter.set_scientific(True)
                            formatter.set_powerlimits((-2, 2))
                            cbar.formatter = formatter
                            cbar.update_ticks()

                            cbar.ax.yaxis.get_offset_text().set_fontsize(14)

                        # Column 4: Error - with its own range
                        error = np.abs(
                            timestep_preds_np[t][0, c] - timestep_targets_np[t][0, c]
                        )

                        # Use actual min and max values for error colormap
                        err_vmin = np.min(error)
                        err_vmax = np.max(error)

                        im = axes[t, 3].imshow(
                            error,
                            cmap="viridis",
                            vmin=err_vmin,
                            vmax=err_vmax,
                        )
                        axes[t, 3].axis("off")
                        cbar = fig.colorbar(im, ax=axes[t, 3], fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=14)

                        if ch_name == "A":
                            formatter = ScalarFormatter(useMathText=True)
                            formatter.set_scientific(True)
                            formatter.set_powerlimits((-2, 2))
                            cbar.formatter = formatter
                            cbar.update_ticks()

                            cbar.ax.yaxis.get_offset_text().set_fontsize(14)

                    plt.tight_layout()

                    # Add column labels (a, b, c, d) above each column
                    col_labels = ["(a)", "(b)", "(c)", "(d)"]
                    for j, label in enumerate(col_labels):
                        ax = axes[0, j]
                        # Get the axis position and place label above it
                        bbox = ax.get_position()
                        # Calculate center of the axis
                        center_x = (bbox.x0 + bbox.x1) / 2
                        # Place label just above the top of the axis
                        label_y = bbox.y1 + 0.02

                        fig.text(
                            center_x,
                            label_y,
                            label,
                            ha="center",
                            va="center",
                            fontsize=22,
                            fontweight="bold",
                        )

                    plt.savefig(
                        os.path.join(
                            plots_dir,
                            f"sim_{sim_idx}_channel_{c}_{ch_name}_publication.png",
                        ),
                        dpi=600,
                        bbox_inches="tight",
                    )

                    plt.savefig(
                        os.path.join(
                            plots_dir,
                            f"sim_{sim_idx}_channel_{c}_{ch_name}_publication.pdf",
                        ),
                        dpi=600,
                        bbox_inches="tight",
                    )

                    plt.close(fig)

                print(
                    f"Publication plots for {split_name} sample {sim_idx} saved to {plots_dir}"
                )

        print(f"All publication plots for {split_name} split saved to {plots_dir}")
        return plots_dir

    except Exception as e:
        print(f"Warning: Could not create publication plots: {e}")
        import traceback

        traceback.print_exc()
        return None
