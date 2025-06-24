#!/usr/bin/env python
"""
This script is being provided as an example and not fully tested.

Generate publication-quality plots using a direct extension of enhanced_diffusion_visualizer.
"""

import os
import sys
import argparse
import torch

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.diffusion import create_diffusion_model
from src.data import get_diffusion_dataloaders
from src.training.train_utils import load_checkpoint
from src.visualization.publication_diffusion_visualizer import (
    save_publication_diffusion_plots,
)


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

    torch.backends.cudnn.benchmark = True

    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    if args.checkpoint:
        checkpoint_path = args.checkpoint
        print(f"Using checkpoint path from command line: {checkpoint_path}")
    else:
        train_params = config["train_params"]
        checkpoint_path = train_params.get("ckpt_path", "")
        print(f"Using checkpoint path from config: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Creating diffusion model from config...")
    model = create_diffusion_model(config)
    model.to(device)

    print(f"Loading checkpoint from {checkpoint_path}")
    load_checkpoint(checkpoint_path, model)
    model.eval()

    print("Creating dataloaders...")
    _, val_loader, test_loader = get_diffusion_dataloaders(config, num_workers=1)

    num_samples = None
    if args.samples:
        num_samples = args.samples
    else:
        wandb_config = config.get("wandb_params", {})
        num_samples = min(4, wandb_config.get("wandb_num_plots", 4))

    print(f"Will generate plots for {num_samples} samples")

    denormalize = config.get("train_params", {}).get("denorm_enhanced_vis", True)
    print(f"Using denormalization: {denormalize}")

    splits_to_process = []
    if args.split == "all":
        if val_loader:
            splits_to_process.append(("val", val_loader))
        if test_loader:
            splits_to_process.append(("test", test_loader))
    elif args.split == "val" and val_loader:
        splits_to_process.append(("val", val_loader))
    elif args.split == "test" and test_loader:
        splits_to_process.append(("test", test_loader))
    else:
        print(f"Warning: No valid loader found for split '{args.split}'")
        return

    for split_name, data_loader in splits_to_process:
        print(f"\nProcessing {split_name} split...")

        save_publication_diffusion_plots(
            model,
            data_loader,
            device,
            args.output,
            split_name=split_name,
            num_samples=num_samples,
            denormalize=denormalize,
        )

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
