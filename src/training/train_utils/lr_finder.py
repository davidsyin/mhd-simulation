"""
Learning rate finder implementation.

This utility helps find an optimal learning rate for your model by training
with exponentially increasing learning rates and plotting the loss curve.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy


class LRFinder:
    """Learning rate finder utility.

    Trains the model for a few iterations with exponentially increasing
    learning rates and records the loss values to help identify the
    optimal learning rate.

    Args:
        model: The neural network model
        optimizer: The optimizer used for training
        criterion: Loss function
        device: Device to run on (cpu or cuda)
    """

    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Save original model and optimizer state to restore later
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

        self.best_lr = None
        self.history = {"lr": [], "loss": []}

    def find(self, dataloader, start_lr=1e-7, end_lr=10, num_iter=100, step_mode="exp"):
        """Run the learning rate finder.

        Args:
            dataloader: DataLoader for training data
            start_lr: Starting learning rate
            end_lr: Maximum learning rate to try
            num_iter: Number of iterations to run
            step_mode: How to step the learning rate ('exp' or 'linear')

        Returns:
            tuple: (list of learning rates, list of losses)
        """
        # Reset model and optimizer to initial state
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        # Set the starting learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = start_lr

        # Calculate the LR multiplier based on the step mode
        if step_mode == "exp":
            lr_factor = (end_lr / start_lr) ** (1 / num_iter)
            lr_schedule = lambda x: start_lr * (lr_factor**x)
        else:  # linear
            lr_step = (end_lr - start_lr) / num_iter
            lr_schedule = lambda x: start_lr + lr_step * x

        # Training loop
        self.history = {"lr": [], "loss": []}
        iteration = 0
        data_iter = iter(dataloader)

        while iteration < num_iter:
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Record the loss and learning rate
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            self.history["loss"].append(loss.item())

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr_schedule(iteration)

            iteration += 1

        # Restore the model and optimizer states
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        # Estimate the best learning rate (where loss decreases fastest)
        losses = np.array(self.history["loss"])
        smooth_losses = np.convolve(losses, np.ones(5) / 5, mode="valid")

        # Find the point of steepest decline
        gradients = np.gradient(smooth_losses)
        best_idx = np.argmin(gradients) + 2  # +2 for the smoothing offset
        if best_idx < len(self.history["lr"]):
            self.best_lr = self.history["lr"][best_idx]

        return self.history["lr"], self.history["loss"]

    def plot(self, log_scale=True, suggest_lr=True):
        """Plot the learning rate vs. loss curve.

        Args:
            log_scale (bool): Whether to use log scale for x-axis
            suggest_lr (bool): Whether to indicate the suggested LR

        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["lr"], self.history["loss"])
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")

        if log_scale:
            plt.xscale("log")

        if suggest_lr and self.best_lr:
            plt.axvline(
                x=self.best_lr,
                color="r",
                linestyle="--",
                label=f"Suggested LR: {self.best_lr:.2e}",
            )
            plt.legend()

        return plt.gcf()


# Example usage:
"""
model = create_model(config)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

lr_finder = LRFinder(model, optimizer, criterion, device)
lr_finder.find(train_loader, start_lr=1e-7, end_lr=1.0, num_iter=100)
fig = lr_finder.plot(suggest_lr=True)
fig.savefig('lr_finder_results.png')

# Use the suggested learning rate
for param_group in optimizer.param_groups:
    param_group['lr'] = lr_finder.best_lr
"""
