import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

from ..utils.config import load_config
from ..models.model_factory import create_model
from ..losses import create_loss
from ..optimizers import create_optimizer, create_scheduler
from ..data.dataset import get_dataloaders
from .trainers import (
    train_epoch,
    train_epoch_distributed,
)
from .validators import (
    validate,
    validate_with_denorm_metrics,
)
from .train_utils import save_checkpoint, load_checkpoint
from ..visualization import save_prediction_plots


def train(config_path, data_path):
    """Custom multi-GPU training that avoids TensorLy issues

    Args:
        config_path: Path to configuration file
        data_path: Path to dataset file
    """
    # Set CUDA benchmark for better performance
    torch.backends.cudnn.benchmark = True

    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()

    # Create logs directory
    wandb_config = config.get("wandb_params", {})
    log_dir = wandb_config.get("wandb_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
    print(
        f"\nTensorBoard logs will be saved to {os.path.join(log_dir, 'tensorboard')}\n"
    )

    batch_size = config["train_loader_params"]["batch_size"]

    train_ratio = config["dataset_params"]["num_train"] / (
        config["dataset_params"]["num_train"] + config["dataset_params"]["num_test"]
    )
    seed = config["dataset_params"].get("seed", 42)

    sub_t = config["dataset_params"].get("sub_t", 1)
    sub_x = config["dataset_params"].get("sub_x", 1)

    print(f"Using subsampling factors: sub_t={sub_t}, sub_x={sub_x}")

    # Create normalization config for dataset
    normalization_config = None
    if "normalization_params" in config:
        # Prepare the normalization configuration for dataloaders
        normalization_config = {"normalization_params": config["normalization_params"]}
        print(
            f"Using {config['normalization_params'].get('type', 'unknown')} normalization at dataset level\n"
        )

    print("Creating dataloaders...")
    start_time = time.time()
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path,
        normalization_config=normalization_config,
        batch_size=batch_size * n_gpus,
        num_workers=config["train_loader_params"]["num_workers"],
        train_ratio=train_ratio,
        seed=seed,
        sub_t=sub_t,
        sub_x=sub_x,
    )
    print(f"Dataloaders created in {time.time() - start_time:.2f} seconds")

    print("Checking data shapes from first batch:")
    x_batch, y_batch = next(iter(train_loader))
    print(f"Input batch shape: {x_batch.shape}")
    print(f"Output batch shape: {y_batch.shape}")

    print("Creating model...")
    model = create_model(config)

    model = model.to(device)

    print(
        f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters\n"
    )

    criterion = create_loss(config)

    optimizer_params = config["optimizer_params"]
    optimizer = create_optimizer(model.parameters(), config)
    print(f"Using {optimizer_params.get('optimizer_type')} optimizer\n")

    if optimizer_params.get("use_scheduler", True):
        scheduler = create_scheduler(optimizer, config)
        print(f"Using {optimizer_params.get('scheduler_type')} scheduler\n")
    else:
        # Create a dummy scheduler that doesn't change the learning rate
        from ..optimizers.standard_schedulers import create_dummy_scheduler

        scheduler = create_dummy_scheduler(optimizer, optimizer_params)
        print("Using dummy scheduler (no LR scheduling)")

    # Load checkpoint if specified
    train_params = config["train_params"]
    start_epoch, best_val_loss = 0, float("inf")
    best_denorm_val_loss = float("inf")  # Track best denormalized validation loss

    if train_params.get("load_ckpt") and os.path.exists(train_params["load_ckpt"]):
        print(f"Loading checkpoint from {train_params['load_ckpt']}")
        checkpoint_data = load_checkpoint(
            train_params["load_ckpt"], model, optimizer, scheduler
        )

        # Unpack checkpoint data - now returned as a dict
        start_epoch = checkpoint_data["epoch"]
        best_val_loss = checkpoint_data["loss"]
        best_denorm_val_loss = checkpoint_data.get("denorm_loss", float("inf"))

        start_epoch += 1  # Start from the next epoch

    # Make sure checkpoint directory exists
    os.makedirs(os.path.dirname(train_params["ckpt_path"]), exist_ok=True)

    print("\nStarting training loop...")
    torch.cuda.empty_cache()  # Clear memory before starting training

    for epoch in range(start_epoch, train_params["epochs"]):
        torch.backends.cudnn.benchmark = True
        epoch_start_time = time.time()

        if n_gpus > 1:
            train_loss = train_epoch_distributed(
                model, config, train_loader, optimizer, criterion, n_gpus
            )
        else:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate with both normalized and denormalized metrics
        val_loss, denorm_val_loss = validate_with_denorm_metrics(
            model, val_loader, criterion, device
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Log metrics to console
        epoch_time = time.time() - epoch_start_time
        print(
            f"\nEpoch {epoch} ({epoch_time:.2f}s): Train Loss: {train_loss:.6f}, "
            f"Val Loss (Norm): {val_loss:.6f}, Val Loss (Denorm): {denorm_val_loss:.6f}, "
            f"LR: {scheduler.get_last_lr()[0] if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else optimizer.param_groups[0]['lr']}\n"
        )

        # Log to tensorboard
        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val_normalized", val_loss, epoch)
        tb_writer.add_scalar("Loss/val_denormalized", denorm_val_loss, epoch)
        tb_writer.add_scalar(
            "LearningRate",
            (
                scheduler.get_last_lr()[0]
                if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                else optimizer.param_groups[0]["lr"]
            ),
            epoch,
        )

        # Save prediction plots
        if (
            epoch % wandb_config.get("wandb_plot_freq", 5) == 0
            or epoch == train_params["epochs"] - 1
        ):
            save_prediction_plots(
                model,
                val_loader,
                device,
                epoch,
                log_dir,
                num_samples=wandb_config.get("wandb_num_plots", 2),
            )

        # Save best model based on normalized loss (original behavior)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save checkpoint with both metrics
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                scheduler,
                train_params["ckpt_path"],  # Use original path without suffix
                denorm_loss=denorm_val_loss,
            )
            print(
                f"Saved best model at epoch {epoch} with val_loss {val_loss:.6f} (denorm: {denorm_val_loss:.6f}) under {train_params['ckpt_path']}\n"
            )

            # Also track if this is the best denormalized loss
            if denorm_val_loss < best_denorm_val_loss:
                best_denorm_val_loss = denorm_val_loss

        # If we find a better denormalized loss but not a better normalized loss
        elif denorm_val_loss < best_denorm_val_loss:
            best_denorm_val_loss = denorm_val_loss
            # Log this event but don't save a new checkpoint
            print(
                f"New best denormalized loss at epoch {epoch}: {denorm_val_loss:.6f} (norm: {val_loss:.6f})"
            )

        print("=" * 110)
        print("\n")
        # Explicitly free up memory
        torch.cuda.empty_cache()

    # Evaluate on test set using both metrics
    test_loss, denorm_test_loss = validate_with_denorm_metrics(
        model, test_loader, criterion, device
    )

    print("=" * 40)
    print(f"\nFinal test loss (normalized): {test_loss:.6f}")
    print(f"Final test loss (denormalized): {denorm_test_loss:.6f}")
    print("=" * 40)

    tb_writer.add_scalar("Loss/test_normalized", test_loss, train_params["epochs"])
    tb_writer.add_scalar(
        "Loss/test_denormalized", denorm_test_loss, train_params["epochs"]
    )
    tb_writer.close()
