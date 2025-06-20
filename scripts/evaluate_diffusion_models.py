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
from src.diffusion import create_diffusion_model
from src.data import get_diffusion_dataloaders
from src.training.validators import validate_with_denorm_metrics_diffusion
from src.training.train_utils import load_checkpoint


def evaluate_diffusion_checkpoint(config_path, device):
    """
    Evaluate a trained diffusion model checkpoint on the test dataset

    Args:
        config_path: Path to the config file
        device: Torch device to use

    Returns:
        dict: Dictionary containing the evaluation metrics or None if evaluation failed
    """
    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    if config.get("config_type", "") != "diffusion":
        print(
            f"⚠️ Config file {os.path.basename(config_path)} is not a diffusion model config."
        )
        return None

    ckpt_path = config["train_params"]["ckpt_path"]
    if not check_path_exists(
        ckpt_path, error_msg=f"⚠️ Checkpoint not found: {ckpt_path}"
    ):
        print(f"Skipping evaluation of {os.path.basename(config_path)}")
        return None

    dataset_params = config.get("dataset_params", {})
    test_path = dataset_params.get("test_path")

    if not test_path:
        print(f"⚠️ Test data path not specified in config")
        return None

    if not check_path_exists(
        test_path, error_msg=f"⚠️ Test data file not found: {test_path}"
    ):
        print(f"Skipping evaluation of {os.path.basename(config_path)}")
        return None

    # Extract normalization config for logging
    norm_type = "none"
    if "normalization_params" in config:
        norm_params = config["normalization_params"]
        norm_type = norm_params.get("type", "unknown")
        print(f"Using {norm_type} normalization")

        # Check for stats files
        inputs_stats_file = norm_params.get("inputs_stats_file")
        targets_stats_file = norm_params.get("targets_stats_file")

        if inputs_stats_file:
            print(f"Using input stats file: {inputs_stats_file}")
            if not os.path.exists(inputs_stats_file):
                print(f"⚠️ Input stats file not found: {inputs_stats_file}")

        if targets_stats_file:
            print(f"Using target stats file: {targets_stats_file}")
            if not os.path.exists(targets_stats_file):
                print(f"⚠️ Target stats file not found: {targets_stats_file}")

    print(f"Creating test dataloader from {test_path}")
    try:
        _, _, test_loader = get_diffusion_dataloaders(config, num_workers=1)

        if test_loader is None:
            print(f"⚠️ Failed to create test dataloader")
            return None

    except Exception as e:
        print(f"⚠️ Error creating dataloaders: {e}")
        print(f"Skipping evaluation of {os.path.basename(config_path)}")
        return None

    # Create diffusion model with same config as training
    print("\nCreating diffusion model...")
    model = create_diffusion_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    model = model.to(device)

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

    # Evaluate on test set
    print("\nEvaluating diffusion model on test set...")
    try:
        test_loss, denorm_test_rel_l2, denorm_test_mse = (
            validate_with_denorm_metrics_diffusion(model, test_loader, device)
        )

        print(f"\nTest results for {os.path.basename(config_path)}:")
        print(f"- Test Loss (normalized): {test_loss:.6f}")
        print(f"- Test Rel L2 (denormalized): {denorm_test_rel_l2:.6f}")
        print(f"- Test MSE (denormalized): {denorm_test_mse:.6f}\n")

        reynolds_number = extract_reynolds_number(os.path.basename(config_path))

        # Get model architecture details for logging
        model_params = config.get("model_params", {})
        model_type = model_params.get("model_type", "unknown")
        channels = model_params.get("channels", 0)
        base_dim = model_params.get("base_dim", 0)
        dim_mults = model_params.get("dim_mults", [])
        attention_info = f"{model_params.get('attn_heads', 0)}x{model_params.get('attn_dim_head', 0)}"

        return {
            "config": os.path.basename(config_path),
            "reynolds_number": reynolds_number,
            "model_type": model_type,
            "channels": channels,
            "base_dim": base_dim,
            "dim_mults": str(dim_mults),
            "attention": attention_info,
            "test_loss": test_loss,
            "denorm_test_rel_l2": denorm_test_rel_l2,
            "denorm_test_mse": denorm_test_mse,
            "checkpoint_epoch": checkpoint_data["epoch"],
            "validation_loss": checkpoint_data["loss"],
            "validation_denorm_rel_l2": val_rel_l2,
            "validation_denorm_mse": val_mse,
            "normalization_type": norm_type,
            "checkpoint_path": ckpt_path,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "sigma_data": model_params.get("sigma_data", 0.0),
            "num_sample_steps": model_params.get("num_sample_steps", 0),
        }
    except Exception as e:
        print(f"⚠️ Error during evaluation: {e}")
        print(f"Skipping evaluation of {os.path.basename(config_path)}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained diffusion models")
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
        default="diffusion_evaluation_results.csv",
        help="Output CSV file path",
    )
    parser.add_argument("--gpus", type=str, default="0", help='GPU indices, e.g., "0"')

    args = parser.parse_args()

    if not check_path_exists(args.config_dir, is_directory=True):
        return

    device = setup_device(args.gpus)

    config_files = find_config_files(args.config_dir, "config_diffusion_Re", args.Re)

    if not config_files:
        if args.Re:
            print(f"No config files found for Reynolds numbers: {args.Re}")
        else:
            print(f"No config_diffusion_Re* files found in {args.config_dir}")
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
            result = evaluate_diffusion_checkpoint(config_path, device)
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
