import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from ..utils.config import load_config
from ..models.model_factory import create_model
from ..losses import create_loss
from ..data.dataset import get_dataloaders
from .trainers import (
    train_epoch,
    train_epoch_distributed,
    train_epoch_distributed_size_mismatch_fix,
)
from .validators import validate, validate_size_mismatch_fix
from .train_utils import save_checkpoint, load_checkpoint
from ..visualization import save_prediction_plots


def train(config_path, data_path):
    """Custom multi-GPU training that avoids TensorLy issues

    Args:
        config_path: Path to configuration file
        data_path: Path to dataset file
    """
    # Load config
    config = load_config(config_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get GPU count
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")

    # Create logs directory
    wandb_config = config.get("wandb_params", {})
    log_dir = wandb_config.get("wandb_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard logger
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
    print(f"TensorBoard logs will be saved to {os.path.join(log_dir, 'tensorboard')}")

    # Get original batch size and create dataloaders
    batch_size = config["train_loader_params"]["batch_size"]

    # When using custom multi-GPU, we'll manually split batches
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path,
        batch_size=batch_size * n_gpus,  # Load larger batches that we'll split
        num_workers=config["train_loader_params"]["num_workers"],
        train_ratio=config["dataset_params"]["num_train"]
        / (
            config["dataset_params"]["num_train"] + config["dataset_params"]["num_test"]
        ),
    )

    # Create model
    model = create_model(config)

    # Move model to primary device
    model = model.to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Define loss function using the factory
    criterion = create_loss(config)

    # Define optimizer
    optimizer_params = config["optimizer_params"]
    optimizer = optim.Adam(
        model.parameters(), lr=optimizer_params["lr"], betas=optimizer_params["betas"]
    )

    # Define scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=optimizer_params["milestones"],
        gamma=optimizer_params["gamma"],
    )

    # Load checkpoint if specified
    train_params = config["train_params"]
    start_epoch, best_val_loss = 0, float("inf")

    if train_params.get("load_ckpt") and os.path.exists(train_params["load_ckpt"]):
        print(f"Loading checkpoint from {train_params['load_ckpt']}")
        start_epoch, best_val_loss = load_checkpoint(
            train_params["load_ckpt"], model, optimizer, scheduler
        )
        start_epoch += 1  # Start from the next epoch

    # Make sure checkpoint directory exists
    os.makedirs(os.path.dirname(train_params["ckpt_path"]), exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, train_params["epochs"]):
        # Train with custom multi-GPU implementation
        if n_gpus > 1:
            train_loss = train_epoch_distributed(
                model, config, train_loader, optimizer, criterion, n_gpus
            )
        else:
            # Fallback to original training for single GPU
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate (always on primary GPU)
        val_loss = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Log metrics to console
        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Log to tensorboard
        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)
        tb_writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        # Save prediction plots
        if epoch % wandb_config.get("wandb_plot_freq", 5) == 0:
            save_prediction_plots(
                model,
                val_loader,
                device,
                epoch,
                log_dir,
                num_samples=wandb_config.get("wandb_num_plots", 2),
            )

        # Save checkpoint
        if epoch % train_params.get("ckpt_freq", 10) == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                scheduler,
                f"{train_params['ckpt_path']}.{epoch}",
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, scheduler, train_params["ckpt_path"]
            )
            print(f"Saved best model at epoch {epoch} with val_loss {val_loss:.6f}")

        # Explicitly free up memory
        torch.cuda.empty_cache()

    # Evaluate on test set
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Final test loss: {test_loss:.6f}")

    tb_writer.add_scalar("Loss/test", test_loss, train_params["epochs"])
    tb_writer.close()
