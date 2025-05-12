import os
import argparse
import torch
import matplotlib.pyplot as plt
import sys

# Add the project root to the path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.neurops.model_factory import create_model
from src.losses import create_loss
from src.optimizers import create_optimizer
from src.data.dataset import get_dataloaders
from src.training.train_utils.lr_finder import LRFinder


def find_best_lr(config_path, data_path, num_iter=100, start_lr=1e-7, end_lr=1.0):
    """Find the best learning rate for the model defined in the config file."""
    # Load config
    config = load_config(config_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get dataset parameters
    train_ratio = config["dataset_params"]["num_train"] / (
        config["dataset_params"]["num_train"] + config["dataset_params"]["num_test"]
    )
    seed = config["dataset_params"].get("seed", 42)
    batch_size = config["train_loader_params"]["batch_size"]

    # Create normalization config for dataset
    normalization_config = None
    if "normalization_params" in config:
        normalization_config = {"normalization_params": config["normalization_params"]}
        print(
            f"Using {config['normalization_params'].get('type', 'unknown')} normalization at dataset level\n"
        )

    # Create dataloaders
    train_loader, _, _ = get_dataloaders(
        data_path,
        normalization_config=normalization_config,
        batch_size=batch_size,
        num_workers=config["train_loader_params"]["num_workers"],
        train_ratio=train_ratio,
        seed=seed,
    )

    # Create model
    model = create_model(config)
    model = model.to(device)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters\n"
    )

    # Define loss function
    criterion = create_loss(config)

    # Create optimizer with default learning rate
    optimizer = create_optimizer(model.parameters(), config)

    # Initialize learning rate finder
    lr_finder = LRFinder(model, optimizer, criterion, device)

    # Run the learning rate finder with improved settings
    print(f"Running learning rate finder for up to {num_iter} iterations...")
    print(f"Learning rate range: {start_lr} to {end_lr}")
    lr_finder.find(
        train_loader,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter,
        diverge_th=5.0,  # Stop if loss grows 5x above minimum
        smoothing_factor=0.05,  # Apply smoothing to reduce noise
    )

    # Plot and save results - skip first few points for clearer visualization
    fig = lr_finder.plot(log_scale=True, suggest_lr=True, skip_start=10, skip_end=5)

    # Create directory for saving results
    os.makedirs("lr_finder_results", exist_ok=True)
    # Save both original and zoomed-in versions
    fig.savefig("lr_finder_results/lr_finder_plot.png")

    # Create a zoomed version focused on the region of interest
    if lr_finder.best_lr:
        fig_zoom = plt.figure(figsize=(12, 8))

        # Focus on area around best LR (within factor of 10)
        plt.plot(lr_finder.history["lr"], lr_finder.history["loss"])
        plt.xscale("log")
        plt.xlim(lr_finder.best_lr / 10, min(lr_finder.best_lr * 10, end_lr))
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.axvline(
            x=lr_finder.best_lr,
            color="r",
            linestyle="--",
            label=f"Suggested LR: {lr_finder.best_lr:.2e}",
        )
        plt.title("Learning Rate Finder (Zoomed)")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.legend()
        fig_zoom.savefig("lr_finder_results/lr_finder_plot_zoomed.png")

    # Print suggested learning rate
    best_lr = lr_finder.best_lr
    if best_lr:
        print(f"\nSuggested learning rate: {best_lr:.6e}")

        # Save suggested learning rate to file
        with open("lr_finder_results/best_lr.txt", "w") as f:
            f.write(f"Suggested learning rate: {best_lr:.10e}\n")
            f.write(f"Configuration file: {config_path}\n")
            f.write(
                f"Original learning rate: {config['optimizer_params']['lr']:.10e}\n"
            )

        # Also save a version of the config with updated learning rate
        updated_config = config.copy()
        updated_config["optimizer_params"]["lr"] = best_lr

        # Generate new config filename
        config_dir = os.path.dirname(config_path)
        config_name = os.path.basename(config_path)
        base_name, ext = os.path.splitext(config_name)
        new_config_path = os.path.join(config_dir, f"{base_name}_best_lr{ext}")

        # Save updated config
        from src.utils.config import save_config

        save_config(updated_config, new_config_path)
        print(f"Updated configuration saved to: {new_config_path}")
    else:
        print("Could not determine best learning rate from the data collected.")

    return best_lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the best learning rate for TFNO3d model"
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
        "--num_iter",
        type=int,
        default=1000,
        help="Number of iterations for learning rate finder",
    )
    parser.add_argument(
        "--start_lr",
        type=float,
        default=1e-5,
        help="Starting learning rate",
    )
    parser.add_argument(
        "--end_lr",
        type=float,
        default=1e-1,
        help="Maximum learning rate to try",
    )
    parser.add_argument(
        "--gpus", type=str, default="0", help='GPU indices, e.g., "0,1"'
    )

    args = parser.parse_args()

    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # If data path not provided, extract from config
    if args.data is None:
        config = load_config(args.config)
        args.data = (
            config["dataset_params"]["data_dir"] + config["dataset_params"]["data_file"]
        )

    # Find best learning rate
    best_lr = find_best_lr(
        args.config,
        args.data,
        num_iter=args.num_iter,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
    )
