"""
Training state management utilities.

This module provides a unified way to manage training state including model weights,
optimizer state, learning rate schedulers, metrics history, and early stopping.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class TrainingState:
    """Manages the complete state of a training run.

    Args:
        model: The neural network model
        optimizer: The optimizer used for training
        scheduler: Learning rate scheduler (optional)
        patience (int): Patience for early stopping (optional)
        mode (str): Mode for monitoring ('min' or 'max')
        save_dir (str): Directory to save state and plots
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        patience=None,
        mode="min",
        save_dir="./runs",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.mode = mode

        # Early stopping setup
        self.patience = patience
        self.counter = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False

        # Training history
        self.history = defaultdict(list)
        self.current_epoch = 0

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

    def update(self, metrics, epoch=None):
        """Update training state with metrics from current epoch.

        Args:
            metrics (dict): Dictionary of metrics for current epoch
            epoch (int): Current epoch number (optional)

        Returns:
            bool: True if this is the best model so far
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        # Update history
        for key, value in metrics.items():
            self.history[key].append(value)

        # Get the monitoring metric
        monitor_metric = metrics.get("val_loss", None)
        if monitor_metric is None:
            raise ValueError("Metrics must include 'val_loss' for monitoring")

        # Check if this is the best model so far
        is_best = False
        if self.mode == "min":
            is_best = monitor_metric < self.best_score
        else:  # mode == 'max'
            is_best = monitor_metric > self.best_score

        # Update best score and reset counter if improved
        if is_best:
            self.best_score = monitor_metric
            self.counter = 0
            self.save_checkpoint(is_best=True)
        else:
            self.counter += 1

        # Check early stopping
        if self.patience and self.counter >= self.patience:
            self.early_stop = True

        # Step the scheduler if provided
        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(monitor_metric)
            else:
                self.scheduler.step()

        # Save regular checkpoint
        self.save_checkpoint(is_best=is_best)

        # Save history
        self.save_history()

        return is_best

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint.

        Args:
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": dict(self.history),
            "best_score": self.best_score,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.save_dir, "latest_checkpoint.pt"))

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, "best_checkpoint.pt"))

    def load_checkpoint(self, path):
        """Load checkpoint from file.

        Args:
            path (str): Path to checkpoint file
        """
        if not os.path.exists(path):
            print(f"Checkpoint {path} does not exist")
            return

        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]

        if "history" in checkpoint:
            self.history = defaultdict(list, checkpoint["history"])

        if "best_score" in checkpoint:
            self.best_score = checkpoint["best_score"]

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def save_history(self):
        """Save training history to file."""
        history_path = os.path.join(self.save_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(dict(self.history), f)

        # Generate plots
        self.plot_metrics()

    def plot_metrics(self):
        """Generate and save plots of training metrics."""
        # Create separate plots for losses and metrics
        plt.figure(figsize=(12, 8))

        # Plot losses
        for key in self.history:
            if "loss" in key.lower():
                plt.plot(self.history[key], label=key)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "losses.png"))
        plt.close()

        # Plot learning rate
        if "lr" in self.history:
            plt.figure(figsize=(10, 6))
            plt.plot(self.history["lr"])
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.yscale("log")
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, "learning_rate.png"))
            plt.close()

        # Plot other metrics
        metrics_to_plot = [
            k for k in self.history if "loss" not in k.lower() and k != "lr"
        ]
        if metrics_to_plot:
            plt.figure(figsize=(12, 8))
            for key in metrics_to_plot:
                plt.plot(self.history[key], label=key)

            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.title("Training Metrics")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, "metrics.png"))
            plt.close()


# Example usage in a training loop:
"""
# Setup
model = create_model(config)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
state = TrainingState(model, optimizer, scheduler, patience=15, save_dir='./runs/experiment1')

# Training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(...)
    val_loss = validate(...)
    
    # Update metrics and check early stopping
    metrics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'lr': optimizer.param_groups[0]['lr']
    }
    
    is_best = state.update(metrics)
    
    if state.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break
"""
