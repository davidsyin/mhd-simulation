#!/usr/bin/env python
"""
Generate publication-quality prediction plots from a trained model.

This script loads a trained model, creates high-quality plots suitable for publication,
and handles special cases such as high Reynolds number simulations correctly.
It infers data paths and checkpoint paths directly from the provided config file.

Usage:
    python generate_publication_plots_neurops.py --config CONFIG_PATH 
        --output OUTPUT_DIR [--split train val test] 
        [--re REYNOLDS_NUMBER] [--samples NUM_SAMPLES]
"""

import os
import re
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Add the project root to the path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neurops.model_factory import create_model
from src.utils.config import load_config
from src.data import get_dataloaders
from src.utils import (
    get_dataset_normalizer,
    identify_data_channels,
    apply_denormalization,
)
from src.training.train_utils import load_checkpoint


def save_publication_quality_plots(model, data_loader, device, output_dir, split_name="val", num_samples=2, reynolds_number=None):
    """Save publication-quality prediction plots to disk

    Args:
        model: Neural network model
        data_loader: DataLoader for the selected data split
        device: Device to run model on
        output_dir: Directory to save plots
        split_name: Name of the data split (train, val, test)
        num_samples: Number of samples to plot
        reynolds_number: Reynolds number of the dataset (if known)
    """
    model.eval()

    plots_dir = os.path.join(output_dir, f"{split_name}_plots")
    os.makedirs(plots_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
    })

    normalizer = get_dataset_normalizer(data_loader)

    with torch.no_grad():
        try:
            x, y = next(iter(data_loader))
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

            # Determine if this is a high Reynolds number case
            high_reynolds = False
            if reynolds_number is not None and reynolds_number >= 750:
                high_reynolds = True
                print(f"High Reynolds number detected (Re={reynolds_number}). Using special colormap limits for vector potential.")

            for i in range(min(num_samples, x.shape[0])):
                fig, axes = plt.subplots(3, 4, figsize=(20, 15))
                
                cmap = "RdBu_r"

                # First column: initial conditions
                for c in range(data_channels):
                    im = axes[c, 0].imshow(x_np[i, c, 0], cmap=cmap)
                    axes[c, 0].axis("off")
                    
                    cbar = fig.colorbar(im, ax=axes[c, 0], fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=14)  # No bold, just larger size

                    # Add scientific notation for the vector potential (A) channel
                    if c == 2:  # For channel A
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()
                        
                        # Make just the size larger for scientific notation
                        cbar.ax.yaxis.get_offset_text().set_fontsize(14)

                # Process remaining columns
                for c in range(data_channels):
                    # Make sure target shape is compatible
                    time_dim = min(y_np.shape[2] - 1, -1)  # Use last time step
                    pred_time_dim = min(pred_np.shape[2] - 1, time_dim)

                    # Set colorbar limits based on true final state
                    vmin = np.min(y_np[i, c, time_dim])
                    vmax = np.max(y_np[i, c, time_dim])

                    # For high Reynolds number and vector potential (A) channel, set separate limits
                    if high_reynolds and c == 2:
                        pred_vmin = np.min(pred_np[i, c, pred_time_dim])
                        pred_vmax = np.max(pred_np[i, c, pred_time_dim])
                    else:
                        pred_vmin, pred_vmax = vmin, vmax

                    # True final state - Column (b)
                    im = axes[c, 1].imshow(
                        y_np[i, c, time_dim], cmap=cmap, vmin=vmin, vmax=vmax
                    )
                    axes[c, 1].axis("off")
                    
                    cbar = fig.colorbar(im, ax=axes[c, 1], fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=14)  # No bold, just larger size

                    # Use scientific notation for the third channel (A)
                    if c == 2:  # For channel A
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()
                        
                        # Size only for scientific notation
                        cbar.ax.yaxis.get_offset_text().set_fontsize(14)

                    # Predicted final state - Column (c)
                    im = axes[c, 2].imshow(
                        pred_np[i, c, pred_time_dim], cmap=cmap, vmin=pred_vmin, vmax=pred_vmax
                    )
                    axes[c, 2].axis("off")

                    cbar = fig.colorbar(im, ax=axes[c, 2], fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=14)  # No bold, just larger size
                        
                    if c == 2:  # For channel A
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()
                        
                        # Size only for scientific notation
                        cbar.ax.yaxis.get_offset_text().set_fontsize(14)

                    # Absolute error - Column (d)
                    error = np.abs(pred_np[i, c, pred_time_dim] - y_np[i, c, time_dim])
                    err_vmin = np.min(error)
                    err_vmax = np.max(error)

                    im = axes[c, 3].imshow(
                        error, cmap="viridis", vmin=err_vmin, vmax=err_vmax
                    )
                    axes[c, 3].axis("off")

                    cbar = fig.colorbar(im, ax=axes[c, 3], fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=14)  # No bold, just larger size
                        
                    if c == 2:  # For channel A
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-2, 2))
                        cbar.formatter = formatter
                        cbar.update_ticks()
                        
                        # Size only for scientific notation
                        cbar.ax.yaxis.get_offset_text().set_fontsize(14)
                
                plt.tight_layout()
                
                col_labels = ["(a)", "(b)", "(c)", "(d)"]
                
                # Use the first row axes to determine column positions
                for j, label in enumerate(col_labels):
                    ax = axes[0, j]
                    # Get the axis position and place label above it
                    bbox = ax.get_position()
                    # Calculate center of the axis
                    center_x = (bbox.x0 + bbox.x1) / 2
                    # Place label just above the top of the axis
                    label_y = bbox.y1 + 0.02
                    
                    fig.text(center_x, label_y, label, 
                             ha='center', va='center', 
                             fontsize=22, fontweight='bold')
                
                
                plt.savefig(os.path.join(plots_dir, f"sample_{i}_publication.png"), dpi=600, bbox_inches='tight')
                print(f"Publication-quality plot for {split_name} split, sample {i} has been saved to {plots_dir}")

                # Also save PDF version for publication
                plt.savefig(os.path.join(plots_dir, f"sample_{i}_publication.pdf"), dpi=600, bbox_inches='tight')
                
                plt.close(fig)

        except RuntimeError as e:
            print(f"Warning: Could not create plots for {split_name} split: {e}")
            import traceback
            traceback.print_exc()
            return


def detect_reynolds_number(config):
    """Attempt to detect Reynolds number from config"""
    reynolds_number = None
    
    dataset_params = config.get("dataset_params", {})
    data_dir = dataset_params.get("data_dir", "")
    data_file = dataset_params.get("data_file", "")
    
    data_path = os.path.join(data_dir, data_file)
    data_path_lower = data_path.lower()
    re_match = re.search(r're(\d+)', data_path_lower)
    if re_match:
        reynolds_number = int(re_match.group(1))
        print(f"Detected Reynolds number from data path: Re={reynolds_number}")
        return reynolds_number
        
    for section in config.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, str) and "re" in value.lower():
                    re_match = re.search(r're(\d+)', value.lower())
                    if re_match:
                        reynolds_number = int(re_match.group(1))
                        print(f"Detected Reynolds number from config value: Re={reynolds_number}")
                        return reynolds_number
    
    print("Could not detect Reynolds number automatically.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality plots from a trained model")
    parser.add_argument("--config", required=True, help="Path to model configuration file")
    parser.add_argument("--output", required=True, help="Directory to save plots")
    parser.add_argument("--checkpoint", help="Path to model checkpoint (overrides config)")
    parser.add_argument("--data", help="Path to dataset file (overrides config)")
    parser.add_argument("--split", nargs='+', 
                        choices=["train", "val", "test", "all"], 
                        default=["val"], 
                        help="Data split(s) to use for plotting (can specify multiple, e.g. --split train test)")
    parser.add_argument("--re", type=int, help="Reynolds number (if not specified in paths)")
    parser.add_argument("--samples", type=int, default=2, help="Number of samples to plot")
    args = parser.parse_args()
    
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    dataset_params = config["dataset_params"]
    data_dir = dataset_params.get("data_dir", "")
    data_file = dataset_params.get("data_file", "")
    
    if args.data:
        data_path = args.data
        print(f"Using data path from command line: {data_path}")
    else:
        data_path = os.path.join(data_dir, data_file)
        print(f"Using data path from config: {data_path}")
        
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        print(f"Using checkpoint path from command line: {checkpoint_path}")
    else:
        train_params = config["train_params"]
        checkpoint_path = train_params.get("ckpt_path", "")
        print(f"Using checkpoint path from config: {checkpoint_path}")
    
    if not os.path.exists(data_path):
        print(f"Warning: Data file not found at {data_path}. Please check the path.")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file not found at {checkpoint_path}. Please check the path.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Creating model from config...")
    model = create_model(config)
    model = model.to(device)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    load_checkpoint(checkpoint_path, model)
    
    normalization_config = None
    if "normalization_params" in config:
        normalization_config = {"normalization_params": config["normalization_params"]}
    
    print(f"Loading data from {data_path}")
    train_ratio = dataset_params["num_train"] / (
        dataset_params["num_train"] + dataset_params["num_test"]
    )
    seed = dataset_params.get("seed", 42)
    sub_t = dataset_params.get("sub_t", 1)
    sub_x = dataset_params.get("sub_x", 1)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path,
        normalization_config=normalization_config,
        batch_size=1,  # Always use batch size 1 for visualization
        num_workers=1,
        train_ratio=train_ratio,
        seed=seed,
        sub_t=sub_t,
        sub_x=sub_x,
    )
    
    # Determine Reynolds number
    reynolds_number = args.re
    if reynolds_number is None:
        reynolds_number = detect_reynolds_number(config)
    
    # Determine which splits to process
    splits_to_process = []
    if "all" in args.split:
        splits_to_process = ["train", "val", "test"]
    else:
        splits_to_process = args.split
    
    print(f"Generating publication-quality plots for splits: {', '.join(splits_to_process)}...")
    
    # Process each requested split
    for split_name in splits_to_process:
        print(f"\nProcessing {split_name} split...")
        
        if split_name == "train":
            data_loader = train_loader
        elif split_name == "val":
            data_loader = val_loader
        elif split_name == "test":
            data_loader = test_loader
            
        save_publication_quality_plots(
            model, data_loader, device, args.output, 
            split_name=split_name, num_samples=args.samples, reynolds_number=reynolds_number
        )
    
    print("\nPlot generation complete!")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = False
    main()
