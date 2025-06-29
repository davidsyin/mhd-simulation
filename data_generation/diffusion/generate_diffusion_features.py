import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(parent_dir)
sys.path.insert(0, project_root)

from src.utils.config import load_config
from src.neurops.model_factory import create_model
from src.data import get_dataloaders
from src.training.train_utils import load_checkpoint

from src.utils import (
    get_dataset_normalizer,
    identify_data_channels,
    apply_denormalization,
)


def run_inference(re_number):
    """Run inference for a specific Reynolds number and save results.

    Args:
        re_number (int): Reynolds number (100, 250, or 500)
    """
    print(f"\n{'=' * 80}")
    print(f"Running inference for Re{re_number}")
    print(f"{'=' * 80}\n")

    config_path = os.path.join(parent_dir, f"configs/config_tfno3d_Re{re_number}.yaml")
    checkpoint_path = os.path.join(
        parent_dir,
        f"checkpoints/permanent_models/tfno_Re{re_number}_6layer_8mod_subt4_minmax.pt",
    )
    output_dir = f"/projects/begd/skacmaz/NODs_diffusion_data/Re{re_number}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    data_path = (
        config["dataset_params"]["data_dir"] + config["dataset_params"]["data_file"]
    )
    print(f"Data will be loaded from: {data_path}")

    print("Creating model from configuration")
    model = create_model(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    model = model.to(device)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint_data = load_checkpoint(checkpoint_path, model)
    print(
        f"Checkpoint loaded from epoch {checkpoint_data['epoch']} with loss {checkpoint_data['loss']:.6f}"
    )

    model.eval()

    print("Creating dataloaders")
    train_size = config["dataset_params"]["num_train"]
    val_plus_test_size = config["dataset_params"]["num_test"]
    seed = config["dataset_params"].get("seed", 42)
    sub_t = config["dataset_params"].get("sub_t", 1)
    sub_x = config["dataset_params"].get("sub_x", 1)

    normalization_config = None
    if "normalization_params" in config:
        normalization_config = {"normalization_params": config["normalization_params"]}
        print(
            f"Using {config['normalization_params'].get('type', 'unknown')} normalization"
        )

    # Use batch size 4 for MI100 GPU
    batch_size = 4

    train_loader, val_loader, test_loader = get_dataloaders(
        data_path,
        normalization_config=normalization_config,
        batch_size=batch_size,
        num_workers=4,
        train_size=train_size,
        val_plus_test_size=val_plus_test_size,
        seed=seed,
        sub_t=sub_t,
        sub_x=sub_x,
    )

    # Dictionary of dataloaders for processing
    dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

    # Process each split separately
    for split_name, loader in dataloaders.items():
        print(f"\nProcessing {split_name} dataset ({len(loader)} batches)")

        all_targets = []
        all_predictions = []

        normalizer = get_dataset_normalizer(loader)
        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx + 1}/{len(loader)}")

            inputs = inputs.to(device)

            data_channel_indices, has_grid_embeddings, _ = identify_data_channels(
                inputs, targets
            )

            with torch.no_grad():
                preds = model(inputs)

            _, targets_denorm, preds_denorm = apply_denormalization(
                inputs,
                targets,
                preds,
                normalizer,
                data_channel_indices,
                has_grid_embeddings,
            )

            all_targets.append(targets_denorm.cpu().numpy())
            all_predictions.append(preds_denorm.cpu().numpy())

            # Free memory
            del inputs, targets, preds, targets_denorm, preds_denorm
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Concatenate results
        print(f"Concatenating {len(all_targets)} batches for {split_name} dataset")
        targets_np = np.concatenate(all_targets, axis=0)
        predictions_np = np.concatenate(all_predictions, axis=0)

        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"{split_name}.npy")
        print(f"Saving {split_name} results to {output_file}")

        results_dict = {"diff_targets": targets_np, "diff_inputs": predictions_np}

        np.save(output_file, results_dict)

        print(
            f"Saved {split_name} data with shapes: targets {targets_np.shape}, predictions {predictions_np.shape}"
        )

        # Free memory
        del all_targets, all_predictions, targets_np, predictions_np
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\nInference for Re{re_number} completed successfully!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference for a specific Reynolds number"
    )
    parser.add_argument(
        "--re",
        type=int,
        choices=[100, 250, 500],
        required=True,
        help="Reynolds number (100, 250, or 500)",
    )

    args = parser.parse_args()
    run_inference(args.re)
