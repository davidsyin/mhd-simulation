import os
import argparse
import torch

from src.utils.config import load_config
from src.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TFNO3d model for MHD simulation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/u/skacmaz/NODs/configs/config_tfno3d.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data file",
    )
    parser.add_argument(
        "--gpus", type=str, default="0", help='GPU indices, e.g., "0,1"'
    )

    args = parser.parse_args()

    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Print device information
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(
            f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Load configuration
    config = load_config(args.config)
    if args.data is None:
        args.data = (
            config["dataset_params"]["data_dir"] + config["dataset_params"]["data_file"]
        )

    # Print model configuration
    print("\nModel Configuration:")
    for key, value in config["model_params"].items():
        print(f"{key}: {value}")

    # Run training
    train(args.config, args.data)
