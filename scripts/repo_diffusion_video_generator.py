"""
Publication-quality video generation functions for diffusion models.
For the easiest usage, add this under src/visualization and run the checkpointed training for a single epoch.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import ScalarFormatter


def save_publication_diffusion_videos(
    model,
    val_loader,
    device,
    output_dir,
    split_name="val",
    num_samples=4,
    denormalize=True,
    fps=7,
    save_vector_potential_only=True,
):
    """
    Generate publication-quality videos of diffusion model predictions showing temporal evolution.

    This creates high-quality videos showing the simultaneous evolution of all three channels
    (u_x, u_y, A) over time, with proper labels, colorbars, and Reynolds number in the title.

    Optionally creates a separate video focusing only on the magnetic vector potential (A) channel.

    Args:
        model: Diffusion model
        val_loader: Validation data loader (with batch_size = timesteps * channels)
        device: Device to run model on
        output_dir: Directory to save videos
        split_name: Name of the data split (val, test)
        num_samples: Number of simulations to create videos for
        denormalize: Whether to denormalize for visualization
        fps: Frames per second for the video
        save_vector_potential_only: Whether to also save a video with only the A channel
    """
    model.eval()

    videos_dir = os.path.join(output_dir, f"{split_name}_publication_videos")
    os.makedirs(videos_dir, exist_ok=True)

    dataset = val_loader.dataset
    config = getattr(dataset, "config", {})
    model_params = config.get("model_params", {})
    dataset_params = config.get("dataset_params", {})

    # Get Reynolds number from config file name or config
    re_number = dataset_params.get("re", None)
    if re_number is None:
        re_number = config.get("re", None)

    # If still not found, try to extract from config file name
    if re_number is None:
        config_file = config.get("config_file", "")
        if not config_file:
            # Try to get from dataset config if available
            config_file = getattr(dataset, "config_file", "")

        import re as regex

        re_match = regex.search(r"_Re(\d+)", config_file)
        if re_match:
            re_number = int(re_match.group(1))
        else:
            re_number = "Unknown"

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

                # Get all timesteps for this simulation
                all_inputs = []
                all_targets = []
                all_predictions = []

                print(f"Processing {timesteps} timesteps for simulation {sim_idx}...")

                for t in range(timesteps):
                    inp = inputs[t].to(device)
                    target = targets[t].to(device)

                    # Add batch dimension if needed for the model
                    if len(inp.shape) == 3:  # If shape is (c, nx, ny)
                        inp = inp.unsqueeze(0)
                    if len(target.shape) == 3:
                        target = target.unsqueeze(0)

                    # Generate prediction
                    pred = model.sample(inp)

                    all_inputs.append(inp)
                    all_targets.append(target)
                    all_predictions.append(pred)

                print(f"Generated predictions for all {len(all_predictions)} timesteps")

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

                        all_inputs_denorm = []
                        all_targets_denorm = []
                        all_preds_denorm = []

                        for t in range(timesteps):
                            all_inputs_denorm.append(
                                denormalize_with_channel_mapping(
                                    all_inputs[t],
                                    input_normalizer,
                                    channel_indices,
                                )
                            )

                            all_targets_denorm.append(
                                denormalize_with_channel_mapping(
                                    all_targets[t],
                                    target_normalizer,
                                    channel_indices,
                                )
                            )

                            all_preds_denorm.append(
                                denormalize_with_channel_mapping(
                                    all_predictions[t],
                                    target_normalizer,
                                    channel_indices,
                                )
                            )

                    except Exception as e:
                        print(f"Warning: Could not denormalize: {e}")
                        print("Using normalized data instead.")
                        all_inputs_denorm = all_inputs
                        all_targets_denorm = all_targets
                        all_preds_denorm = all_predictions
                else:
                    all_inputs_denorm = all_inputs
                    all_targets_denorm = all_targets
                    all_preds_denorm = all_predictions

                # Convert to numpy arrays
                all_inputs_np = [t.cpu().numpy() for t in all_inputs_denorm]
                all_targets_np = [t.cpu().numpy() for t in all_targets_denorm]
                all_preds_np = [t.cpu().numpy() for t in all_preds_denorm]

                n_channels = all_inputs_np[0].shape[1]

                print(
                    f"Data shapes: inputs={len(all_inputs_np)}, targets={len(all_targets_np)}, predictions={len(all_preds_np)}"
                )
                print(
                    f"Expected timesteps: {timesteps}, actual data timesteps: {len(all_inputs_np)}"
                )

                # Create video showing all three channels side by side
                # Layout: 3 rows x 3 columns
                # Top row: PINO-Only Prediction (u_x, u_y, A)
                # Middle row: Ground Truth (u_x, u_y, A)
                # Bottom row: Framework Final Prediction (u_x, u_y, A)
                fig, axes = plt.subplots(3, 3, figsize=(18, 18))

                # Set up the figure with proper spacing
                plt.subplots_adjust(
                    left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.3, hspace=0.3
                )

                # Add main title with Reynolds number
                fig.suptitle(
                    f"Spatiotemporal Evolution - Re = {re_number}",
                    fontsize=20,
                    fontweight="bold",
                    y=0.96,
                )

                # Initialize plots
                ims = []
                cbars = []

                # Calculate global color limits for consistent scaling across time
                target_mins = []
                target_maxs = []
                input_mins = []
                input_maxs = []

                for c in range(n_channels):
                    # Get global min/max for targets (ground truth and predictions will use same range)
                    target_data = np.array(
                        [all_targets_np[t][0, c] for t in range(timesteps)]
                    )
                    target_mins.append(np.min(target_data))
                    target_maxs.append(np.max(target_data))

                    # Get global min/max for inputs (PINO-only predictions)
                    input_data = np.array(
                        [all_inputs_np[t][0, c] for t in range(timesteps)]
                    )
                    input_mins.append(np.min(input_data))
                    input_maxs.append(np.max(input_data))

                cmap = "RdBu_r"

                # Initialize the first frame
                for c in range(n_channels):
                    ch_name = (
                        channel_mappings[c]
                        if c < len(channel_mappings)
                        else f"Channel {c}"
                    )

                    # Top row: PINO-Only Prediction
                    im_input = axes[0, c].imshow(
                        all_inputs_np[0][0, c],
                        cmap=cmap,
                        vmin=input_mins[c],
                        vmax=input_maxs[c],
                        animated=True,
                    )

                    axes[0, c].set_title(
                        f"PINO-Only Prediction ({ch_name})",
                        fontsize=16,
                        fontweight="bold",
                    )
                    axes[0, c].set_xticks([])
                    axes[0, c].set_yticks([])

                    # Add colorbar for PINO prediction
                    cbar_input = fig.colorbar(
                        im_input, ax=axes[0, c], fraction=0.046, pad=0.04
                    )
                    cbar_input.ax.tick_params(labelsize=12)

                    # Scientific notation for vector potential
                    if ch_name == "A":
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar_input.formatter = formatter
                        cbar_input.update_ticks()
                        cbar_input.ax.yaxis.get_offset_text().set_fontsize(12)

                    # Middle row: Ground Truth
                    im_target = axes[1, c].imshow(
                        all_targets_np[0][0, c],
                        cmap=cmap,
                        vmin=target_mins[c],
                        vmax=target_maxs[c],
                        animated=True,
                    )

                    axes[1, c].set_title(
                        f"Ground Truth ({ch_name})", fontsize=16, fontweight="bold"
                    )
                    axes[1, c].set_xticks([])
                    axes[1, c].set_yticks([])

                    # Add colorbar for ground truth
                    cbar_target = fig.colorbar(
                        im_target, ax=axes[1, c], fraction=0.046, pad=0.04
                    )
                    cbar_target.ax.tick_params(labelsize=12)

                    # Scientific notation for vector potential
                    if ch_name == "A":
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar_target.formatter = formatter
                        cbar_target.update_ticks()
                        cbar_target.ax.yaxis.get_offset_text().set_fontsize(12)

                    # Bottom row: Framework Final Prediction
                    im_pred = axes[2, c].imshow(
                        all_preds_np[0][0, c],
                        cmap=cmap,
                        vmin=target_mins[c],
                        vmax=target_maxs[c],
                        animated=True,
                    )

                    axes[2, c].set_title(
                        f"Framework Final Prediction ({ch_name})",
                        fontsize=16,
                        fontweight="bold",
                    )
                    axes[2, c].set_xticks([])
                    axes[2, c].set_yticks([])

                    # Add colorbar for framework prediction
                    cbar_pred = fig.colorbar(
                        im_pred, ax=axes[2, c], fraction=0.046, pad=0.04
                    )
                    cbar_pred.ax.tick_params(labelsize=12)

                    # Scientific notation for vector potential
                    if ch_name == "A":
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar_pred.formatter = formatter
                        cbar_pred.update_ticks()
                        cbar_pred.ax.yaxis.get_offset_text().set_fontsize(12)

                    ims.extend([im_input, im_target, im_pred])

                def animate(frame):
                    """Update function for animation"""
                    # Always log frame for debugging
                    print(f"Processing frame {frame}/{timesteps - 1}")

                    # Ensure frame index is within bounds
                    if frame >= len(all_inputs_np):
                        print(
                            f"Warning: Frame {frame} exceeds available data length {len(all_inputs_np)}"
                        )
                        frame = len(all_inputs_np) - 1

                    # Update all channel plots
                    for c in range(n_channels):
                        # Update PINO-only prediction (top row)
                        ims[c * 3].set_array(all_inputs_np[frame][0, c])

                        # Update ground truth (middle row)
                        ims[c * 3 + 1].set_array(all_targets_np[frame][0, c])

                        # Update framework final prediction (bottom row)
                        ims[c * 3 + 2].set_array(all_preds_np[frame][0, c])

                    return ims

                # Create animation
                print(f"Creating animation with {timesteps} frames...")
                # Use a list of frame numbers to ensure all frames are included
                frame_list = list(range(timesteps))
                print(
                    f"Frame sequence: {frame_list[:10]}...{frame_list[-5:] if len(frame_list) > 10 else frame_list}"
                )

                anim = animation.FuncAnimation(
                    fig,
                    animate,
                    frames=frame_list,
                    interval=1000 // fps,
                    blit=False,
                    repeat=False,
                )

                # Save video in high quality
                video_path = os.path.join(
                    videos_dir, f"sim_{sim_idx}_evolution_Re{re_number}.mp4"
                )

                print(
                    f"Saving video for simulation {sim_idx} with {timesteps} frames at {fps} fps..."
                )

                # Use high quality settings with explicit codec parameters
                Writer = animation.writers["ffmpeg"]
                writer = Writer(
                    fps=fps,
                    metadata=dict(artist="NODs"),
                    bitrate=5000,
                    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
                )

                anim.save(video_path, writer=writer, dpi=150)

                plt.close(fig)

                print(
                    f"Publication video for {split_name} sample {sim_idx} saved to {video_path}"
                )

                # Create vector potential only video if requested
                if save_vector_potential_only and n_channels >= 3:
                    # Find the A channel index
                    a_channel_idx = None
                    for i, ch_name in enumerate(channel_mappings):
                        if ch_name == "A":
                            a_channel_idx = i
                            break

                    if a_channel_idx is not None:
                        print(
                            f"Creating vector potential only video for simulation {sim_idx}..."
                        )

                        # Create 1x3 layout for vector potential only
                        fig_a, axes_a = plt.subplots(1, 3, figsize=(18, 6))
                        plt.subplots_adjust(
                            left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.3
                        )

                        # Add main title
                        fig_a.suptitle(
                            f"Magnetic Vector Potential Evolution - Re = {re_number}",
                            fontsize=20,
                            fontweight="bold",
                            y=0.95,
                        )

                        # Get A channel data limits
                        a_target_min = target_mins[a_channel_idx]
                        a_target_max = target_maxs[a_channel_idx]
                        a_input_min = input_mins[a_channel_idx]
                        a_input_max = input_maxs[a_channel_idx]

                        # Initialize plots for A channel
                        im_a_input = axes_a[0].imshow(
                            all_inputs_np[0][0, a_channel_idx],
                            cmap=cmap,
                            vmin=a_input_min,
                            vmax=a_input_max,
                            animated=True,
                        )
                        axes_a[0].set_title(
                            "PINO-Only Prediction", fontsize=16, fontweight="bold"
                        )
                        axes_a[0].set_xticks([])
                        axes_a[0].set_yticks([])

                        cbar_a_input = fig_a.colorbar(
                            im_a_input, ax=axes_a[0], fraction=0.046, pad=0.04
                        )
                        cbar_a_input.ax.tick_params(labelsize=12)
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar_a_input.formatter = formatter
                        cbar_a_input.update_ticks()
                        cbar_a_input.ax.yaxis.get_offset_text().set_fontsize(12)

                        im_a_target = axes_a[1].imshow(
                            all_targets_np[0][0, a_channel_idx],
                            cmap=cmap,
                            vmin=a_target_min,
                            vmax=a_target_max,
                            animated=True,
                        )
                        axes_a[1].set_title(
                            "Ground Truth", fontsize=16, fontweight="bold"
                        )
                        axes_a[1].set_xticks([])
                        axes_a[1].set_yticks([])

                        cbar_a_target = fig_a.colorbar(
                            im_a_target, ax=axes_a[1], fraction=0.046, pad=0.04
                        )
                        cbar_a_target.ax.tick_params(labelsize=12)
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar_a_target.formatter = formatter
                        cbar_a_target.update_ticks()
                        cbar_a_target.ax.yaxis.get_offset_text().set_fontsize(12)

                        im_a_pred = axes_a[2].imshow(
                            all_preds_np[0][0, a_channel_idx],
                            cmap=cmap,
                            vmin=a_target_min,
                            vmax=a_target_max,
                            animated=True,
                        )
                        axes_a[2].set_title(
                            "Framework Final Prediction", fontsize=16, fontweight="bold"
                        )
                        axes_a[2].set_xticks([])
                        axes_a[2].set_yticks([])

                        cbar_a_pred = fig_a.colorbar(
                            im_a_pred, ax=axes_a[2], fraction=0.046, pad=0.04
                        )
                        cbar_a_pred.ax.tick_params(labelsize=12)
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar_a_pred.formatter = formatter
                        cbar_a_pred.update_ticks()
                        cbar_a_pred.ax.yaxis.get_offset_text().set_fontsize(12)

                        def animate_a(frame):
                            """Update function for A channel animation"""
                            if (
                                frame % 5 == 0 or frame == timesteps - 1
                            ):  # Only print every 5th frame + last frame
                                print(
                                    f"Animating A channel frame {frame}/{timesteps - 1}"
                                )

                            # Ensure frame index is within bounds
                            if frame >= len(all_inputs_np):
                                print(
                                    f"Warning: A channel frame {frame} exceeds available data length {len(all_inputs_np)}"
                                )
                                frame = len(all_inputs_np) - 1

                            im_a_input.set_array(all_inputs_np[frame][0, a_channel_idx])
                            im_a_target.set_array(
                                all_targets_np[frame][0, a_channel_idx]
                            )
                            im_a_pred.set_array(all_preds_np[frame][0, a_channel_idx])
                            return [im_a_input, im_a_target, im_a_pred]

                        # Create A channel animation
                        print(
                            f"Creating A channel animation with {timesteps} frames..."
                        )
                        anim_a = animation.FuncAnimation(
                            fig_a,
                            animate_a,
                            frames=timesteps,
                            interval=1000 // fps,
                            blit=False,
                            repeat=True,
                        )

                        # Save A channel video
                        video_path_a = os.path.join(
                            videos_dir,
                            f"sim_{sim_idx}_vector_potential_Re{re_number}.mp4",
                        )

                        print(
                            f"Saving A channel video for simulation {sim_idx} with {timesteps} frames at {fps} fps..."
                        )

                        Writer = animation.writers["ffmpeg"]
                        writer_a = Writer(
                            fps=fps, metadata=dict(artist="NODs"), bitrate=5000
                        )
                        anim_a.save(video_path_a, writer=writer_a, dpi=150)

                        plt.close(fig_a)
                        print(
                            f"Vector potential video for {split_name} sample {sim_idx} saved to {video_path_a}"
                        )
                    else:
                        print(
                            f"Warning: Vector potential channel (A) not found in channel mappings for sample {sim_idx}"
                        )

        print(f"All publication videos for {split_name} split saved to {videos_dir}")
        return videos_dir

    except Exception as e:
        print(f"Warning: Could not create publication videos: {e}")
        import traceback

        traceback.print_exc()
        return None
