import os
import argparse
import torch

from src.utils.config import load_config
from src.training import train_neural_operator, train_diffusion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TFNO3d model for MHD simulation or diffusion model"
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

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(
            f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    config = load_config(args.config)
    if args.data is None:
        args.data = (
            config["dataset_params"]["data_dir"] + config["dataset_params"]["data_file"]
        )

    print("\nModel Configuration:")
    for key, value in config["model_params"].items():
        print(f"{key}: {value}")

    # Check if this is a diffusion model or neural operator based on config
    if config.get("config_type", "neural_operator") == "diffusion":
        print("\nTraining Diffusion Model...")
        train_diffusion(args.config, args.data)
    else:
        print("\nTraining Neural Operator...")
        train_neural_operator(args.config, args.data)
