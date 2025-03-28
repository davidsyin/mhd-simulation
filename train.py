import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter

from model import PhysicsNormalizedTFNO
from dataset import get_dataloaders


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """Create model from config"""
    model_params = config["model_params"]

    # Create model
    model = PhysicsNormalizedTFNO(
        modes1=model_params.get("modes1", 8),
        modes2=model_params.get("modes2", 8),
        modes3=model_params.get("modes3", 8),
        width=model_params.get("width", 8),  # Using width=8 as default
        n_layers=model_params.get("n_layers", 4),
        in_dim=model_params.get("in_dim", 3),
        out_dim=model_params.get("out_dim", 3),
        fc_dim=model_params.get("fc_dim", 128),
        activation=model_params.get("activation", "gelu"),
        pad_x=model_params.get("pad_x", 0.1),
        pad_y=model_params.get("pad_y", 0.1),
        pad_z=model_params.get("pad_z", 0.1),
        input_norm=model_params.get("input_norm", [1.0, 1.0, 1.0]),
        output_norm=model_params.get("output_norm", [1.0, 1.0, 1.0]),
        rank=model_params.get("rank", 0.5),
        factorization=model_params.get("factorization", "cp"),
    )

    return model


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader)
    for x, y in pbar:
        # Move to device
        x, y = x.to(device), y.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        pred = model(x)

        # Calculate loss
        loss = criterion(pred, y)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Log loss
        total_loss += loss.item()
        pbar.set_description(f"Train Loss: {loss.item():.6f}")

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            # Move to device
            x, y = x.to(device), y.to(device)

            # Forward pass
            pred = model(x)

            # Calculate loss
            loss = criterion(pred, y)

            # Log loss
            total_loss += loss.item()

    return total_loss / len(val_loader)


def save_checkpoint(model, optimizer, epoch, loss, scheduler, ckpt_path):
    """Save checkpoint"""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "loss": loss,
        },
        ckpt_path,
    )


def load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None):
    """Load checkpoint"""
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} does not exist. Starting from scratch.")
        return 0, float("inf")

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]


def save_prediction_plots(model, val_loader, device, epoch, output_dir, num_samples=2):
    """Save prediction plots to disk instead of using wandb"""
    model.eval()

    plots_dir = os.path.join(output_dir, "prediction_plots", f"epoch_{epoch}")
    os.makedirs(plots_dir, exist_ok=True)

    with torch.no_grad():
        # Get a batch of validation data
        x, y = next(iter(val_loader))
        x, y = x.to(device), y.to(device)

        # Forward pass
        pred = model(x)

        # Convert to numpy for plotting
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        pred_np = pred.cpu().numpy()

        # Plot predictions
        for i in range(min(num_samples, x.shape[0])):
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))

            # Channel names for better interpretability
            channel_names = ["u_x", "u_y", "A"]

            # First row: initial conditions
            for c in range(3):
                axes[0, c].imshow(x_np[i, c, 0])
                axes[0, c].set_title(f"IC: {channel_names[c]}")
                axes[0, c].axis("off")

            # Second row: true final state
            for c in range(3):
                axes[1, c].imshow(y_np[i, c, -1])
                axes[1, c].set_title(f"True Final: {channel_names[c]}")
                axes[1, c].axis("off")

            # Third row: predicted final state
            for c in range(3):
                axes[2, c].imshow(pred_np[i, c, -1])
                axes[2, c].set_title(f"Pred Final: {channel_names[c]}")
                axes[2, c].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"sample_{i}.png"))
            plt.close(fig)


def train(config_path, data_path):
    """Main training function without wandb dependency"""
    # Load config
    config = load_config(config_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create logs directory
    wandb_config = config.get("wandb_params", {})
    log_dir = wandb_config.get("wandb_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard logger
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
    print(f"TensorBoard logs will be saved to {os.path.join(log_dir, 'tensorboard')}")

    # Create dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path,
        batch_size=config["train_loader_params"]["batch_size"],
        num_workers=config["train_loader_params"]["num_workers"],
        train_ratio=config["dataset_params"]["num_train"]
        / (
            config["dataset_params"]["num_train"] + config["dataset_params"]["num_test"]
        ),
    )

    # Create model
    model = create_model(config)
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Define loss function based on config
    criterion = nn.MSELoss()

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
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
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

    # Evaluate on test set
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Final test loss: {test_loss:.6f}")

    tb_writer.add_scalar("Loss/test", test_loss, train_params["epochs"])
    tb_writer.close()
