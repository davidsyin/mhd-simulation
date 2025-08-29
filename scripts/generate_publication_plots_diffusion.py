#!/usr/bin/env python
"""
This script is being provided as an example and not fully tested.

Generate publication-quality plots using a direct extension of enhanced_diffusion_visualizer.
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.diffusion import create_diffusion_model
from src.data import get_diffusion_dataloaders
from src.training.train_utils import load_checkpoint
from src.visualization.enhanced_diffusion_visualizer import (
    save_enhanced_diffusion_plots,
)
from publication_diffusion_visualizer import save_publication_diffusion_plots


def get_splits_to_process(split_arg, val_loader, test_loader):
    """
    Determine which data splits to process based on command line argument.

    Args:
        split_arg: Command line split argument ("val", "test", or "all")
        val_loader: Validation data loader
        test_loader: Test data loader

    Returns:
        List of (split_name, data_loader) tuples to process

    Raises:
        ValueError: If requested split is not available
    """
    available_splits = {
        "val": ("validation", val_loader),
        "test": ("test", test_loader),
    }

    if split_arg == "all":
        splits_to_process = [
            split_data
            for split_data in available_splits.values()
            if split_data[1] is not None
        ]
        if not splits_to_process:
            raise ValueError("No data loaders available")
        return splits_to_process
    else:
        if split_arg in available_splits and available_splits[split_arg][1] is not None:
            return [available_splits[split_arg]]
        else:
            raise ValueError(f"Split '{split_arg}' not available or loader is None")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots from diffusion model"
    )
    parser.add_argument(
        "--config", required=True, help="Path to model configuration file"
    )
    parser.add_argument("--output", required=True, help="Directory to save plots")
    parser.add_argument(
        "--checkpoint", help="Path to model checkpoint (overrides config)"
    )
    parser.add_argument(
        "--split",
        choices=["val", "test", "all"],
        default="val",
        help="Data split to use for plotting",
    )
    parser.add_argument("--samples", type=int, help="Number of samples to plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output, exist_ok=True)

    print("\nLoading diffusion model configuration...")
    config = load_config(args.config)

    print("\nModel Configuration:")
    for key, value in config["model_params"].items():
        print(f"{key}: {value}")

    print("\nCreating diffusion dataloaders...")
    train_loader, val_loader, test_loader = get_diffusion_dataloaders(
        config, num_workers=config.get("train_loader_params", {}).get("num_workers", 1)
    )

    if val_loader:
        x_batch, y_batch = next(iter(val_loader))
        print(f"Input batch shape: {x_batch.shape}")
        print(f"Output batch shape: {y_batch.shape}\n")

    print("\nCreating diffusion model...")
    model = create_diffusion_model(config)
    model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters\n")

    train_params = config["train_params"]
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        print(f"Using checkpoint path from command line: {checkpoint_path}")
    else:
        checkpoint_path = train_params["ckpt_path"]
        print(f"Using checkpoint path from config: {checkpoint_path}")

    print(f"\nLoading checkpoint from {checkpoint_path}")
    checkpoint_data = load_checkpoint(checkpoint_path, model)

    print(
        f"Checkpoint loaded from epoch {checkpoint_data['epoch']} with validation loss {checkpoint_data['loss']:.6f}\n"
    )

    try:
        splits_to_process = get_splits_to_process(args.split, val_loader, test_loader)
    except ValueError as e:
        print(f"Error: {e}")
        return

    for split_name, data_loader in splits_to_process:
        if data_loader is None:
            print(f"No {split_name} loader available. Skipping.")
            continue

        print(f"\nProcessing {split_name} set...")
        split_output_dir = os.path.join(args.output, split_name)
        os.makedirs(split_output_dir, exist_ok=True)

        print(f"\nGenerating publication diffusion visualizations...")
        save_publication_diffusion_plots(
            model,
            data_loader,
            device,
            split_output_dir,
            split_name=split_name,
            num_samples=args.samples if args.samples else 1,
            denormalize=train_params.get(
                "denorm_enhanced_vis", True
            ),  # Use training setting
        )
        print(f"Publication diffusion plots saved to {split_output_dir}")

    print(f"\nAll processing complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
