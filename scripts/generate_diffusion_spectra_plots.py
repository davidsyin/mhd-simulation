#!/usr/bin/env python
"""
Script to generate both spatial visualizations and spectral plots from a trained diffusion model.

This script closely mimics the training pipeline's structure to ensure compatibility,
using the exact same dataloader setup, model loading, and visualization techniques.

Usage:
    python scripts/generate_diffusion_visuals.py --config configs/config_diffusion_Re500.yaml
                                              --spectra_dir ./spectra_results
                                              --output_dir ./diffusion_visuals
"""

import os
import argparse
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator, NullFormatter
import time
import pickle
import sys
import copy
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.diffusion import create_diffusion_model
from src.data import get_diffusion_dataloaders
from src.training.train_utils import load_checkpoint
from src.visualization.spectra_plots import compute_energy_spectra
from src.visualization.enhanced_diffusion_visualizer import (
    save_enhanced_diffusion_plots,
)


def setup_mpl_for_publication():
    """Configure matplotlib for publication-quality figures suitable for Physics Letters B."""
    mpl.rcParams.update(
        {
            # Font settings for physics publication
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 14,  # Increased from 12
            "mathtext.fontset": "cm",
            # Figure settings
            "figure.figsize": (12, 10),  # Bigger figure size
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Axes settings for clear visualization
            "axes.linewidth": 1.0,  # Thicker axes
            "axes.labelsize": 22,  # Bigger labels
            "axes.titlesize": 24,  # Bigger title
            # Tick settings
            "xtick.major.width": 1.0,  # Thicker ticks
            "ytick.major.width": 2.0,
            "xtick.major.size": 7,  # Slightly longer ticks
            "ytick.major.size": 7,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.labelsize": 18,  # Bigger tick labels
            "ytick.labelsize": 18,
            # Legend settings
            "legend.fontsize": 16,  # Much bigger legend text
            "legend.framealpha": 0.85,
            "legend.edgecolor": "0.8",
            "legend.loc": "best",
            # Line settings
            "lines.linewidth": 3.0,  # Much thicker lines
            "lines.markersize": 10,
        }
    )


def generate_combined_spectra_plot(
    k,
    target_spectra_kin,
    target_spectra_mag,
    neural_op_spectra_kin,
    neural_op_spectra_mag,
    diffusion_spectra_kin,
    diffusion_spectra_mag,
    output_path,
    use_log_scale=True,
    fig_size=(10, 8),
):
    """
    Generate a publication-quality spectral plot with both neural operator and diffusion model predictions.

    Args:
        k: Wave numbers
        target_spectra_kin/mag: True kinetic/magnetic energy spectra
        neural_op_spectra_kin/mag: Neural operator predicted kinetic/magnetic energy spectra
        diffusion_spectra_kin/mag: Diffusion model predicted kinetic/magnetic energy spectra
        output_path: Path to save the plot
        use_log_scale: Whether to use log scale
        fig_size: Figure size
    """
    # Find appropriate axis limits
    xmin = None
    for idx, (e_k_true, e_m_true) in enumerate(
        zip(target_spectra_kin, target_spectra_mag)
    ):
        if e_k_true > 1e-8 or e_m_true > 1e-8:
            xmin = max(k[idx] * 0.9, k[0])
            break
    if xmin is None or (use_log_scale and xmin <= 0):
        xmin = max(0.1, k[0])

    # Find min/max energy values for y-axis
    min_energy = float("inf")
    max_energy = 0
    for spectra in [
        target_spectra_kin,
        target_spectra_mag,
        neural_op_spectra_kin,
        neural_op_spectra_mag,
        diffusion_spectra_kin,
        diffusion_spectra_mag,
    ]:
        for val in spectra:
            if val > 0 and val < min_energy:
                min_energy = val
            if val > max_energy:
                max_energy = val

    ymin = min_energy * 0.5 if min_energy != float("inf") else 1e-10
    if use_log_scale and ymin <= 0:
        ymin = 1e-10

    ymax = max_energy * 2.0  # Give more room at top for legends

    # Create figure with high quality settings
    plt.figure(figsize=fig_size)

    # Target colors (UIUC official colors)
    target_kin_color = "#13294B"  # UIUC Blue for target kinetic
    target_mag_color = "#FF552E"  # UIUC Orange for target magnetic

    # PINO+Diffusion colors (vibrant with contrast to targets)
    # Using a cross-palette approach for intuitive comparison.
    diffusion_kin_color = "#FF552E"  # A strong, dark-ish Red to contrast with the Blue
    # diffusion_kin_color = "#c22d2d"  # A strong, dark-ish Red to contrast with the Blue
    diffusion_mag_color = (
        "#13294B"  # A confident, medium Blue to contrast with the Orange
    )
    # diffusion_mag_color = (
    #     "#9467bd"  # A confident, medium Blue to contrast with the Orange
    # )

    # PINO-Only colors (muted versions of diffusion colors)
    # Desaturated and lighter versions of the vibrant pair above.
    neural_op_kin_color = "#b3d2b5"  # Muted/dusty version of the Red
    neural_op_mag_color = "#a5b3d2"  # Muted/dusty version of the Blue

    # Plot with appropriate scale and improved styling for physics publication
    if use_log_scale:
        # Target = solid lines, PINO+Diffusion = dashed, PINO-Only = dotted
        plt.loglog(
            k,
            target_spectra_kin,
            color=target_kin_color,
            linestyle="-",
            label=r"$E_{\mathrm{kin}}$ (Target)",
        )
        plt.loglog(
            k,
            target_spectra_mag,
            color=target_mag_color,
            linestyle="-",
            label=r"$E_{\mathrm{mag}}$ (Target)",
        )

        plt.loglog(
            k,
            diffusion_spectra_kin,
            color=diffusion_kin_color,
            linestyle="--",
            label=r"$E_{\mathrm{kin}}$ (PINO+Diffusion)",
        )
        plt.loglog(
            k,
            diffusion_spectra_mag,
            color=diffusion_mag_color,
            linestyle="--",
            label=r"$E_{\mathrm{mag}}$ (PINO+Diffusion)",
        )

        plt.loglog(
            k,
            neural_op_spectra_kin,
            color=neural_op_kin_color,
            linestyle=":",
            label=r"$E_{\mathrm{kin}}$ (PINO-Only)",
        )
        plt.loglog(
            k,
            neural_op_spectra_mag,
            color=neural_op_mag_color,
            linestyle=":",
            label=r"$E_{\mathrm{mag}}$ (PINO-Only)",
        )
    else:
        plt.plot(
            k,
            target_spectra_kin,
            color=target_kin_color,
            linestyle="-",
            label=r"$E_{\mathrm{kin}}$ (Target)",
        )
        plt.plot(
            k,
            target_spectra_mag,
            color=target_mag_color,
            linestyle="-",
            label=r"$E_{\mathrm{mag}}$ (Target)",
        )

        plt.plot(
            k,
            diffusion_spectra_kin,
            color=diffusion_kin_color,
            linestyle="--",
            label=r"$E_{\mathrm{kin}}$ (PINO+Diffusion)",
        )
        plt.plot(
            k,
            diffusion_spectra_mag,
            color=diffusion_mag_color,
            linestyle="--",
            label=r"$E_{\mathrm{mag}}$ (PINO+Diffusion)",
        )

        plt.plot(
            k,
            neural_op_spectra_kin,
            color=neural_op_kin_color,
            linestyle=":",
            label=r"$E_{\mathrm{kin}}$ (PINO-Only)",
        )
        plt.plot(
            k,
            neural_op_spectra_mag,
            color=neural_op_mag_color,
            linestyle=":",
            label=r"$E_{\mathrm{mag}}$ (PINO-Only)",
        )

    # Improve axes labels with LaTeX formatting - standard for physics publications
    plt.xlabel(r"$k$", fontsize=20)
    plt.ylabel(r"$E(k)$", fontsize=20)

    # Set x limit to 110 as requested
    plt.xlim(left=xmin, right=110)
    plt.ylim(bottom=ymin, top=ymax)

    # Format tick labels - remove y-ticks as requested
    if use_log_scale:
        ax = plt.gca()
        # Configure x-axis
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10)
        )

        # Remove y-tick labels but keep the gridlines
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
        # Remove y-tick labels
        plt.gca().yaxis.set_major_formatter(NullFormatter())
        plt.gca().yaxis.set_minor_formatter(NullFormatter())

    # Add grid for better readability - common in spectral analysis
    plt.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.3)
    if use_log_scale:
        plt.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.15)

    # Move legend to bottom left
    legend = plt.legend(
        loc="lower left",
        framealpha=0.85,
        edgecolor="0.8",
        fancybox=True,
        ncol=2,
        fontsize=16,
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    # Save the plot with high resolution required for publication
    plt.savefig(output_path, dpi=600, bbox_inches="tight", format="png")

    # Also save as PDF for vector graphics (often required by journals)
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight", format="pdf")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate diffusion model visualizations and spectral analysis"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to diffusion model config file"
    )
    parser.add_argument(
        "--spectra_dir",
        type=str,
        required=True,
        help="Directory containing neural operator spectral data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./diffusion_visuals",
        help="Directory to save outputs",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to process"
    )

    args = parser.parse_args()

    # Set CUDA device - matching the training pipeline setup
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Enable cuDNN benchmark for better performance (matches training setting)
    torch.backends.cudnn.benchmark = True

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up matplotlib for publication-quality figures
    setup_mpl_for_publication()

    # Load config - exactly as done in training
    print("\nLoading diffusion model configuration...")
    config = load_config(args.config)

    print("\nModel Configuration:")
    for key, value in config["model_params"].items():
        print(f"{key}: {value}")

    # Create dataloaders - exactly as done in training
    print("\nCreating diffusion dataloaders...")
    train_loader, val_loader, test_loader = get_diffusion_dataloaders(
        config, num_workers=config.get("train_loader_params", {}).get("num_workers", 1)
    )

    # Print data shapes - as done during training
    if val_loader:
        x_batch, y_batch = next(iter(val_loader))
        print(f"Input batch shape: {x_batch.shape}")
        print(f"Output batch shape: {y_batch.shape}\n")

    # Create model - exactly as in training
    print("\nCreating diffusion model...")
    model = create_diffusion_model(config)
    model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters\n")

    # Load checkpoint - as in training
    train_params = config["train_params"]
    checkpoint_path = train_params["ckpt_path"]
    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint_data = load_checkpoint(checkpoint_path, model)

    print(
        f"Checkpoint loaded from epoch {checkpoint_data['epoch']} with validation loss {checkpoint_data['loss']:.6f}\n"
    )

    # Process validation and test sets
    for split_name, data_loader in [("validation", val_loader), ("test", test_loader)]:
        if data_loader is None:
            print(f"No {split_name} loader available. Skipping.")
            continue

        print(f"\nProcessing {split_name} set...")
        split_output_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)

        # 1. Generate enhanced diffusion visualizations - EXACTLY as in training
        print(f"\nGenerating enhanced diffusion visualizations...")
        enhanced_plots_dir = save_enhanced_diffusion_plots(
            model,
            data_loader,
            device,
            f"inference",
            split_output_dir,
            num_samples=args.num_samples,
            denormalize=train_params.get(
                "denorm_enhanced_vis", True
            ),  # Use training setting
        )
        print(f"Enhanced diffusion plots saved to {enhanced_plots_dir}")

        # 2. Generate spectral plots if spectra directory exists
        try:
            # Load the neural operator spectral data
            spectra_filepath = os.path.join(
                args.spectra_dir, split_name, f"{split_name}_spectra_data.pkl"
            )
            with open(spectra_filepath, "rb") as f:
                neural_op_spectra = pickle.load(f)

            print(f"Loaded neural operator spectral data from {spectra_filepath}")

            # Extract wave numbers and spectral data
            k = neural_op_spectra["k"]
            target_spectra_kin = neural_op_spectra["true_spectra_kin"]
            target_spectra_mag = neural_op_spectra["true_spectra_mag"]
            neural_op_spectra_kin = neural_op_spectra["pred_spectra_kin"]
            neural_op_spectra_mag = neural_op_spectra["pred_spectra_mag"]
            dx = neural_op_spectra.get("dx", 1.0)
            dy = neural_op_spectra.get("dy", 1.0)

            # Get first sample (full simulation)
            with torch.no_grad():
                # Reset data loader iterator to ensure we start from the beginning
                data_iter = iter(data_loader)
                inputs, targets = next(data_iter)

                # Get only the last timestep from the batch
                input_last = inputs[-1].unsqueeze(0).to(device)
                target_last = targets[-1].unsqueeze(0).to(device)

                print(f"Generating diffusion model prediction for the last timestep...")
                # Make a prediction for the last timestep (exactly as done in training)
                diffusion_pred = model.sample(input_last)

                # Calculate spectral data for the diffusion prediction
                print("Calculating spectral data for diffusion prediction...")
                # Get normalization info directly from the dataset
                dataset = data_loader.dataset
                input_normalizer = getattr(dataset, "input_normalizer", None)
                target_normalizer = getattr(dataset, "target_normalizer", None)

                # If needed, denormalize before computing spectra (to match physical units)
                if target_normalizer is not None and hasattr(
                    target_normalizer, "denormalize"
                ):
                    print("Denormalizing prediction before computing spectra...")
                    # Handle potential custom denormalization needs
                    try:
                        # Get channel indices (if any)
                        channel_indices = config.get("dataset_params", {}).get(
                            "channel_indices", None
                        )
                        if isinstance(channel_indices, int):
                            channel_indices = [channel_indices]
                        elif channel_indices is None:
                            channel_indices = list(
                                range(min(diffusion_pred.shape[1], 3))
                            )

                        # For denormalization, try to follow dataset's approach
                        if hasattr(dataset, "prepare_diffusion_input"):
                            # This will handle specific channel indices if needed
                            def denorm_with_appropriate_channels(
                                tensor, normalizer, indices
                            ):
                                # Create time dimension expected by normalizer
                                tensor_with_time = tensor.unsqueeze(
                                    2
                                )  # [B, C, 1, H, W]

                                # If we need to handle specific channels
                                if indices is not None:
                                    # Create full 3-channel tensor
                                    bs, c, _, h, w = tensor_with_time.shape
                                    full_tensor = torch.zeros(
                                        bs, 3, 1, h, w, device=tensor.device
                                    )

                                    # Place channels in correct positions
                                    for i, channel_idx in enumerate(indices):
                                        if i < c:
                                            full_tensor[:, channel_idx] = (
                                                tensor_with_time[:, i]
                                            )

                                    # Denormalize the full tensor
                                    denorm_tensor = normalizer.denormalize(full_tensor)

                                    # Extract only the needed channels
                                    result = torch.zeros(
                                        bs, c, h, w, device=tensor.device
                                    )
                                    for i, channel_idx in enumerate(indices):
                                        if i < c:
                                            result[:, i] = denorm_tensor[
                                                :, channel_idx, 0
                                            ]
                                else:
                                    # Simpler case - just denormalize
                                    denorm_tensor = normalizer.denormalize(
                                        tensor_with_time
                                    )
                                    result = denorm_tensor.squeeze(2)  # Remove time dim

                                return result

                            # Denormalize using our helper
                            diffusion_pred_denorm = denorm_with_appropriate_channels(
                                diffusion_pred, target_normalizer, channel_indices
                            )
                            target_last_denorm = denorm_with_appropriate_channels(
                                target_last, target_normalizer, channel_indices
                            )
                        else:
                            # Fallback to simple denormalization
                            diffusion_pred_denorm = target_normalizer.denormalize(
                                diffusion_pred
                            )
                            target_last_denorm = target_normalizer.denormalize(
                                target_last
                            )

                    except Exception as e:
                        print(
                            f"WARNING: Denormalization failed: {e}. Using normalized data for spectral analysis."
                        )
                        diffusion_pred_denorm = diffusion_pred
                        target_last_denorm = target_last
                else:
                    diffusion_pred_denorm = diffusion_pred
                    target_last_denorm = target_last

                # Compute energy spectra on (potentially) denormalized data
                _, diffusion_spectra_kin, diffusion_spectra_mag = (
                    compute_energy_spectra(diffusion_pred_denorm.cpu(), dx=dx, dy=dy)
                )

                # Debug: Compare data ranges
                print("\nData ranges for debugging:")
                print(
                    f"Target range: [{target_last_denorm.min().item():.6f}, {target_last_denorm.max().item():.6f}]"
                )
                print(
                    f"Diffusion prediction range: [{diffusion_pred_denorm.min().item():.6f}, {diffusion_pred_denorm.max().item():.6f}]"
                )

                for ch_idx in range(min(diffusion_pred_denorm.shape[1], 3)):
                    ch_name = ["u_x", "u_y", "A"][ch_idx]
                    print(f"Channel {ch_idx} ({ch_name}):")
                    print(
                        f"  Target: min={target_last_denorm[:, ch_idx].min().item():.6f}, "
                        f"max={target_last_denorm[:, ch_idx].max().item():.6f}"
                    )
                    print(
                        f"  Prediction: min={diffusion_pred_denorm[:, ch_idx].min().item():.6f}, "
                        f"max={diffusion_pred_denorm[:, ch_idx].max().item():.6f}"
                    )

                # Generate and save the combined plot
                plot_path = os.path.join(
                    split_output_dir, f"{split_name}_combined_spectra.png"
                )
                print(f"Generating combined spectral plot and saving to {plot_path}")

                # Use first simulation only from spectral analysis
                generate_combined_spectra_plot(
                    k,
                    target_spectra_kin,
                    target_spectra_mag,
                    neural_op_spectra_kin,
                    neural_op_spectra_mag,
                    diffusion_spectra_kin[0],
                    diffusion_spectra_mag[0],
                    plot_path,
                    True,
                    (10, 8),
                )

                # Save the combined data
                combined_data = {
                    "k": k,
                    "target_spectra_kin": target_spectra_kin,
                    "target_spectra_mag": target_spectra_mag,
                    "neural_op_spectra_kin": neural_op_spectra_kin,
                    "neural_op_spectra_mag": neural_op_spectra_mag,
                    "diffusion_spectra_kin": diffusion_spectra_kin[0],
                    "diffusion_spectra_mag": diffusion_spectra_mag[0],
                    "dx": dx,
                    "dy": dy,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                combined_data_path = os.path.join(
                    split_output_dir, f"{split_name}_combined_spectra_data.pkl"
                )
                with open(combined_data_path, "wb") as f:
                    pickle.dump(combined_data, f)
                print(f"Saved combined spectral data to {combined_data_path}")

        except Exception as e:
            print(f"Error processing spectral data for {split_name} set: {e}")
            import traceback

            traceback.print_exc()

    print(f"\nAll processing complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
