"""Functions for saving and loading model checkpoints during training."""

import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def save_checkpoint(
    model, optimizer, epoch, loss, scheduler, ckpt_path, denorm_loss=None
):
    """Save checkpoint

    Args:
        model: Neural network model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        scheduler: Learning rate scheduler
        ckpt_path: Path to save checkpoint
        denorm_loss: Denormalized loss value (optional)
    """
    # If using DDP, save only the model's module
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }

    # Add denormalized loss if provided
    if denorm_loss is not None:
        checkpoint_data["denorm_loss"] = denorm_loss

    torch.save(checkpoint_data, ckpt_path)


def load_checkpoint(ckpt_path, model, optimizer=None, scheduler=None):
    """Load checkpoint

    Args:
        ckpt_path: Path to checkpoint
        model: Neural network model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)

    Returns:
        dict: Checkpoint data with epoch, loss, and denorm_loss if available
    """
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} does not exist. Starting from scratch.")
        return {"epoch": 0, "loss": float("inf"), "denorm_loss": float("inf")}

    checkpoint = torch.load(ckpt_path, weights_only=False)

    # Clean up the model state dict if _metadata is present
    if "_metadata" in checkpoint["model_state_dict"]:
        del checkpoint["model_state_dict"]["_metadata"]

    # Handle DDP model
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Return all checkpoint data as a dictionary
    result = {
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"],
        "denorm_loss": checkpoint.get("denorm_loss", float("inf")),
    }

    return result
