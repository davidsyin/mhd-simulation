#!/usr/bin/env python
"""
Script to generate publication-quality spectral plots from trained neural operator models.

This script loads a checkpoint, runs inference on the FIRST sample from validation and test sets,
and generates physics publication-quality spectral analysis plots.

Usage:
    python scripts/generate_neurops_spectra_plots.py --config configs/config.yaml
                                          --checkpoint checkpoints/model.pt
                                          --output_dir ./spectra_results
"""

import os
import argparse
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, AutoMinorLocator
import time
import pickle
from pathlib import Path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.neurops.model_factory import create_model
from src.data import get_dataloaders
from src.training.train_utils import load_checkpoint
from src.visualization.spectra_plots import compute_energy_spectra
from src.utils import (
    get_dataset_normalizer,
    identify_data_channels,
    apply_denormalization,
)


def setup_mpl_for_publication():
    """Configure matplotlib for publication-quality figures suitable for Physics Letters B."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 12,
            "mathtext.fontset": "cm",
            "figure.figsize": (10, 8),
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.linewidth": 1.5,
            "axes.labelsize": 18,
            "axes.titlesize": 20,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "legend.framealpha": 0.8,
            "legend.edgecolor": "0.8",
            "legend.loc": "best",
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
        }
    )


def generate_publication_spectra_plot(
    k,
    true_spectra_kin,
    true_spectra_mag,
    pred_spectra_kin,
    pred_spectra_mag,
    title,
    output_path,
    use_log_scale=True,
    fig_size=(10, 8),
):
    """
    Generate a publication-quality spectral plot suitable for Physics Letters B.

    Args:
        k: Wave numbers
        true_spectra_kin: True kinetic energy spectra
        true_spectra_mag: True magnetic energy spectra
        pred_spectra_kin: Predicted kinetic energy spectra
        pred_spectra_mag: Predicted magnetic energy spectra
        title: Plot title (not used anymore as requested)
        output_path: Path to save the plot
        use_log_scale: Whether to use log scale
        fig_size: Figure size
    """
    # Find appropriate axis limits
    xmin = None
    for idx, (e_k_true, e_m_true) in enumerate(zip(true_spectra_kin, true_spectra_mag)):
        if e_k_true > 1e-8 or e_m_true > 1e-8:
            xmin = max(k[idx] * 0.9, k[0])
            break
    if xmin is None or (use_log_scale and xmin <= 0):
        xmin = max(0.1, k[0])

    # Find min/max energy values for y-axis
    min_energy = float("inf")
    max_energy = 0
    for spectra in [
        true_spectra_kin,
        true_spectra_mag,
        pred_spectra_kin,
        pred_spectra_mag,
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

    plt.figure(figsize=fig_size)

    # Use sophisticated seaborn-inspired color scheme
    target_kin_color = "#4c72b0"  # Deep blue for target kinetic
    target_mag_color = "#55a868"  # Deep green for target magnetic
    pred_kin_color = "#c44e52"  # Brick red for pred kinetic
    pred_mag_color = "#8172b3"  # Purple for pred magnetic

    # Plot with appropriate scale, styles, and improved styling for physics publication
    if use_log_scale:
        # Target = solid lines, Pred = dashed lines
        plt.loglog(
            k,
            true_spectra_kin,
            color=target_kin_color,
            linestyle="-",
            linewidth=2.5,
            label=r"$E_{\mathrm{kin}}$ (Target)",
        )
        plt.loglog(
            k,
            pred_spectra_kin,
            color=pred_kin_color,
            linestyle="--",
            linewidth=2.5,
            label=r"$E_{\mathrm{kin}}$ (Pred)",
        )
        plt.loglog(
            k,
            true_spectra_mag,
            color=target_mag_color,
            linestyle="-",
            linewidth=2.5,
            label=r"$E_{\mathrm{mag}}$ (Target)",
        )
        plt.loglog(
            k,
            pred_spectra_mag,
            color=pred_mag_color,
            linestyle="--",
            linewidth=2.5,
            label=r"$E_{\mathrm{mag}}$ (Pred)",
        )
    else:
        plt.plot(
            k,
            true_spectra_kin,
            color=target_kin_color,
            linestyle="-",
            linewidth=2.5,
            label=r"$E_{\mathrm{kin}}$ (Target)",
        )
        plt.plot(
            k,
            pred_spectra_kin,
            color=pred_kin_color,
            linestyle="--",
            linewidth=2.5,
            label=r"$E_{\mathrm{kin}}$ (Pred)",
        )
        plt.plot(
            k,
            true_spectra_mag,
            color=target_mag_color,
            linestyle="-",
            linewidth=2.5,
            label=r"$E_{\mathrm{mag}}$ (Target)",
        )
        plt.plot(
            k,
            pred_spectra_mag,
            color=pred_mag_color,
            linestyle="--",
            linewidth=2.5,
            label=r"$E_{\mathrm{mag}}$ (Pred)",
        )

    plt.xlabel(r"$k$", fontsize=20)
    plt.ylabel(r"$E(k)$", fontsize=20)

    plt.xlim(left=xmin, right=170)
    plt.ylim(bottom=ymin, top=ymax)

    # Format tick labels for better readability
    if use_log_scale:
        ax = plt.gca()
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        # Add minor ticks
        ax.yaxis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10)
        )
        ax.xaxis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10)
        )
    else:
        plt.gca().xaxis.set_minor_locator(AutoMinorLocator())
        plt.gca().yaxis.set_minor_locator(AutoMinorLocator())

    plt.grid(True, which="major", linestyle="-", linewidth=0.7, alpha=0.3)
    if use_log_scale:
        plt.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.15)

    legend = plt.legend(
        loc="upper right",
        framealpha=0.85,
        edgecolor="0.8",
        fancybox=True,
        ncol=2,
        fontsize=14,
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()

    plt.savefig(output_path, dpi=600, bbox_inches="tight", format="png")

    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches="tight", format="pdf")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality spectral plots from neural operator model"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./spectra_outputs",
        help="Directory to save spectral plots and data",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument(
        "--use_log_scale",
        action="store_true",
        default=True,
        help="Use logarithmic scale for plots",
    )
    parser.add_argument(
        "--fig_size",
        type=float,
        nargs=2,
        default=[10, 8],
        help="Figure size in inches (width, height)",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    setup_mpl_for_publication()

    config = load_config(args.config)
    print("\nModel Configuration:")
    for key, value in config["model_params"].items():
        print(f"{key}: {value}")

    print("\nCreating model...")
    model = create_model(config)

    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint_data = load_checkpoint(args.checkpoint, model)
    model = model.to(device)
    model.eval()

    print(
        f"Checkpoint loaded from epoch {checkpoint_data['epoch']} with validation loss {checkpoint_data['loss']:.6f}"
    )
    if "denorm_loss_mse" in checkpoint_data:
        print(f"Denormalized MSE loss: {checkpoint_data['denorm_loss_mse']:.6f}")
    if "denorm_loss_rel_l2" in checkpoint_data:
        print(
            f"Denormalized relative L2 loss: {checkpoint_data['denorm_loss_rel_l2']:.6f}"
        )

    data_dir = config["dataset_params"]["data_dir"]
    data_file = config["dataset_params"]["data_file"]
    data_path = os.path.join(data_dir, data_file)

    normalization_config = None
    if "normalization_params" in config:
        normalization_config = {"normalization_params": config["normalization_params"]}
        print(
            f"\nUsing {config['normalization_params'].get('type', 'unknown')} normalization"
        )

    sub_t = config["dataset_params"].get("sub_t", 1)
    sub_x = config["dataset_params"].get("sub_x", 1)
    print(f"\nUsing subsampling factors: sub_t={sub_t}, sub_x={sub_x}")

    print("\nCreating dataloaders...")
    start_time = time.time()
    train_size = config["dataset_params"]["num_train"]
    val_plus_test_size = config["dataset_params"]["num_test"]
    seed = config["dataset_params"].get("seed", 42)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_path,
        normalization_config=normalization_config,
        batch_size=1,  # Process one sample at a time for spectral analysis
        num_workers=1,
        train_size=train_size,
        val_plus_test_size=val_plus_test_size,
        seed=seed,
        sub_t=sub_t,
        sub_x=sub_x,
    )
    print(f"Dataloaders created in {time.time() - start_time:.2f} seconds")

    x_batch, y_batch = next(iter(val_loader))
    print(f"\nInput batch shape: {x_batch.shape}")
    print(f"Output batch shape: {y_batch.shape}")

    normalizer = get_dataset_normalizer(val_loader)

    print("\nProcessing FIRST sample from validation set...")
    process_first_sample(
        model,
        val_loader,
        device,
        normalizer,
        args.output_dir,
        "validation",
        args.use_log_scale,
        tuple(args.fig_size),
    )

    print("\nProcessing FIRST sample from test set...")
    process_first_sample(
        model,
        test_loader,
        device,
        normalizer,
        args.output_dir,
        "test",
        args.use_log_scale,
        tuple(args.fig_size),
    )

    print(f"\nAll spectral plots and data saved to {args.output_dir}")


def process_first_sample(
    model,
    dataloader,
    device,
    normalizer,
    output_dir,
    dataset_name,
    use_log_scale,
    fig_size,
):
    """Process only the first sample from a dataset and generate spectral plots."""
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Process only the first sample
    with torch.no_grad():
        # Get the first sample
        try:
            x, y = next(iter(dataloader))
            x, y = x.to(device), y.to(device)

            print(f"Processing first {dataset_name} sample")

            pred = model(x)

            data_channel_indices, has_grid_embeddings, _ = identify_data_channels(x, y)

            # Apply denormalization if needed
            _, y_denorm, pred_denorm = apply_denormalization(
                x, y, pred, normalizer, data_channel_indices, has_grid_embeddings
            )

            # Calculate grid spacing (assuming uniform grid)
            dx, dy = 1.0, 1.0  # Default values
            if has_grid_embeddings:
                try:
                    # Extract grid coordinates
                    grid_x = x[0, 1].unique()  # Assuming 2nd channel is x coordinates
                    grid_y = x[0, 2].unique()  # Assuming 3rd channel is y coordinates

                    # Calculate grid spacing
                    dx = float((grid_x[1] - grid_x[0]).cpu().numpy())
                    dy = float((grid_y[1] - grid_y[0]).cpu().numpy())
                except (IndexError, RuntimeError) as e:
                    print(f"Warning: Could not extract grid spacing: {e}")
                    print("Using default grid spacing dx=dy=1.0")

            # Get last time step for both
            y_last = y_denorm[0, :, -1].cpu()
            pred_last = pred_denorm[0, :, -1].cpu()

            # Calculate spectra
            k, true_spectra_kin, true_spectra_mag = compute_energy_spectra(
                y_last, dx, dy
            )
            _, pred_spectra_kin, pred_spectra_mag = compute_energy_spectra(
                pred_last, dx, dy
            )

            plot_filename = f"{dataset_name}_spectra.png"
            plot_path = os.path.join(dataset_dir, plot_filename)

            generate_publication_spectra_plot(
                k,
                true_spectra_kin[0],
                true_spectra_mag[0],
                pred_spectra_kin[0],
                pred_spectra_mag[0],
                None,
                plot_path,
                use_log_scale,
                fig_size,
            )

            print(f"Saved spectral plot to {plot_path}")

            # Save data for this sample
            sample_data = {
                "k": k,
                "true_spectra_kin": true_spectra_kin[0],
                "true_spectra_mag": true_spectra_mag[0],
                "pred_spectra_kin": pred_spectra_kin[0],
                "pred_spectra_mag": pred_spectra_mag[0],
                "dx": dx,
                "dy": dy,
                "grid_shape": (y_last.shape[1], y_last.shape[2]),
            }

            # Save as pickle file
            pickle_file_path = os.path.join(
                dataset_dir, f"{dataset_name}_spectra_data.pkl"
            )
            with open(pickle_file_path, "wb") as f:
                pickle.dump(sample_data, f)
            print(f"Saved {dataset_name} spectra data to {pickle_file_path}")

            # Also save as NumPy file
            np_file_path = os.path.join(dataset_dir, f"{dataset_name}_spectra.npz")
            np.savez(
                np_file_path,
                k=k,
                true_spectra_kin=true_spectra_kin[0],
                true_spectra_mag=true_spectra_mag[0],
                pred_spectra_kin=pred_spectra_kin[0],
                pred_spectra_mag=pred_spectra_mag[0],
                metadata=np.array(
                    [
                        {
                            "dx": dx,
                            "dy": dy,
                            "grid_shape": (y_last.shape[1], y_last.shape[2]),
                        }
                    ],
                    dtype=object,
                ),
            )

            # Save metadata about this run
            metadata = {
                "dataset": dataset_name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "use_log_scale": use_log_scale,
                "grid_info": {
                    "dx": dx,
                    "dy": dy,
                },
            }

            meta_path = os.path.join(dataset_dir, f"{dataset_name}_metadata.pkl")
            with open(meta_path, "wb") as f:
                pickle.dump(metadata, f)

        except Exception as e:
            print(f"Error processing first {dataset_name} sample: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
