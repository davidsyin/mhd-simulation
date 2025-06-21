import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import traceback

from eval_utils.common import (
    setup_device,
    find_config_files,
    extract_reynolds_number,
    check_path_exists,
)
from eval_utils.reporting import (
    create_output_filename,
    save_results,
    print_summary_table,
)

from src.utils.config import load_config
from src.neurops.model_factory import create_model
from src.losses import create_loss
from src.data import get_dataloaders
from src.training.validators import validate_with_denorm_metrics
from src.training.train_utils import load_checkpoint


def evaluate_checkpoint(config_path, device):
    """
    Evaluate a trained TFNO3D model checkpoint on the test dataset

    Args:
        config_path: Path to the config file
        device: Torch device to use

    Returns:
        dict: Dictionary containing the evaluation metrics or None if evaluation failed
    """
    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    ckpt_path = config["train_params"]["ckpt_path"]
    if not check_path_exists(
        ckpt_path, error_msg=f"⚠️ Checkpoint not found: {ckpt_path}"
    ):
        print(f"Skipping evaluation of {os.path.basename(config_path)}")
        return None

    data_dir = config["dataset_params"].get("data_dir", "")
    data_file = config["dataset_params"].get("data_file", "")
    data_path = os.path.join(data_dir, data_file)

    print(f"Data path: {data_path}")
    if not check_path_exists(
        data_path, error_msg=f"⚠️ Data file not found: {data_path}"
    ):
        print(f"Skipping evaluation of {os.path.basename(config_path)}")
        return None

    # Setup normalization configuration
    normalization_config = None
    norm_type = "none"
    if "normalization_params" in config:
        normalization_config = {"normalization_params": config["normalization_params"]}
        norm_type = config["normalization_params"].get("type", "unknown")
        print(f"Using {norm_type} normalization")

        # Check for stats file
        if "stats_file" in config["normalization_params"]:
            stats_file = config["normalization_params"]["stats_file"]
            print(f"Using stats file: {stats_file}")
            if not os.path.exists(stats_file):
                print(f"⚠️ Stats file not found: {stats_file}")
                print(f"Continuing with potentially incorrect normalization")

    sub_t = config["dataset_params"].get("sub_t", 1)
    sub_x = config["dataset_params"].get("sub_x", 1)

    train_ratio = config["dataset_params"]["num_train"] / (
        config["dataset_params"]["num_train"] + config["dataset_params"]["num_test"]
    )
    seed = config["dataset_params"].get("seed", 42)

    print(f"Creating dataloaders with train_ratio={train_ratio}, seed={seed}")
    print(f"Subsampling factors: sub_t={sub_t}, sub_x={sub_x}")

    try:
        _, _, test_loader = get_dataloaders(
            data_path,
            normalization_config=normalization_config,
            batch_size=1,  # Use batch size 1 for evaluation
            num_workers=1,
            train_ratio=train_ratio,
            seed=seed,
            sub_t=sub_t,
            sub_x=sub_x,
        )
    except Exception as e:
        print(f"⚠️ Error creating dataloaders: {e}")
        print(f"Skipping evaluation of {os.path.basename(config_path)}")
        return None

    # Create model with same config as training
    print("\nCreating model...")
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    model = model.to(device)

    criterion = create_loss(config)

    print(f"Loading checkpoint from {ckpt_path}")
    try:
        checkpoint_data = load_checkpoint(ckpt_path, model)

        print(f"Checkpoint loaded from epoch {checkpoint_data['epoch']}")
        print(f"Best validation loss: {checkpoint_data['loss']:.6f}")
        val_rel_l2 = checkpoint_data.get("denorm_loss_rel_l2", float("inf"))
        val_mse = checkpoint_data.get("denorm_loss_mse", float("inf"))
        if "denorm_loss_rel_l2" in checkpoint_data:
            print(f"Best validation denorm rel L2: {val_rel_l2:.6f}")
        if "denorm_loss_mse" in checkpoint_data:
            print(f"Best validation denorm MSE: {val_mse:.6f}")
    except Exception as e:
        print(f"⚠️ Error loading checkpoint: {e}")
        print(f"Skipping evaluation of {os.path.basename(config_path)}")
        return None

    # Evaluate on test set - this will handle both normalized and denormalized metrics
    print("\nEvaluating on test set...")
    try:
        test_loss, test_rel_l2, test_mse = validate_with_denorm_metrics(
            model, test_loader, criterion, device
        )

        print(f"\nTest results for {os.path.basename(config_path)}:")
        print(f"- Test Loss (normalized): {test_loss:.6f}")
        print(f"- Test Rel L2 : {test_rel_l2:.6f}")
        print(f"- Test MSE : {test_mse:.6f}\n")

        reynolds_number = extract_reynolds_number(os.path.basename(config_path))
        model_variant = ""
        if "_6layer" in os.path.basename(config_path):
            model_variant = "6layer"

        return {
            "config": os.path.basename(config_path),
            "reynolds_number": reynolds_number,
            "model_variant": model_variant,
            "test_loss": test_loss,
            "test_rel_l2": test_rel_l2,
            "test_mse": test_mse,
            "checkpoint_epoch": checkpoint_data["epoch"],
            "validation_loss": checkpoint_data["loss"],
            "validation_denorm_rel_l2": val_rel_l2,
            "validation_denorm_mse": val_mse,
            "sub_t": sub_t,
            "sub_x": sub_x,
            "normalization_type": norm_type,
            "checkpoint_path": ckpt_path,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        }
    except Exception as e:
        print(f"⚠️ Error during evaluation: {e}")
        print(f"Skipping evaluation of {os.path.basename(config_path)}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained TFNO3D models")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="../configs",
        help="Directory containing config files",
    )
    parser.add_argument(
        "--Re",
        type=str,
        default=None,
        help="Comma-separated list of specific Reynolds numbers to evaluate (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tfno3d_evaluation_results.csv",
        help="Output CSV file path",
    )
    parser.add_argument("--gpus", type=str, default="0", help='GPU indices, e.g., "0"')

    args = parser.parse_args()

    if not check_path_exists(args.config_dir, is_directory=True):
        return

    device = setup_device(args.gpus)

    config_files = find_config_files(args.config_dir, "config_tfno3d_Re", args.Re)

    if not config_files:
        if args.Re:
            print(f"No config files found for Reynolds numbers: {args.Re}")
        else:
            print(f"No config_tfno3d_Re* files found in {args.config_dir}")
        return

    print(f"Found {len(config_files)} config files:")
    for cf in config_files:
        print(f"- {os.path.basename(cf)}")

    output_file = create_output_filename(args.output)

    # Evaluate each checkpoint and collect results
    results = []
    for config_path in config_files:
        print(f"\n{'=' * 80}")
        print(f"Processing {os.path.basename(config_path)}")
        print(f"{'=' * 80}")

        try:
            result = evaluate_checkpoint(config_path, device)
            if result:
                results.append(result)
                save_results(results, output_file, incremental=True)
        except Exception as e:
            print(f"Error processing {os.path.basename(config_path)}:")
            print(f"{type(e).__name__}: {e}")
            traceback.print_exc()

    if results:
        save_results(results, output_file)
        print_summary_table(results)
    else:
        print("No results were obtained. Please check the logs for errors.")


if __name__ == "__main__":
    main()
