"""
Learning rate finder implementation.

This utility helps find an optimal learning rate for your model by training
with exponentially increasing learning rates and plotting the loss curve.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.signal import savgol_filter


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

        # Save original model state properly
        state_dict = model.state_dict()
        # Create a clean copy without _metadata
        self.model_state = {}
        for k, v in state_dict.items():
            if k == "_metadata":
                continue
            if isinstance(v, torch.Tensor):
                self.model_state[k] = v.detach().clone()
            else:
                self.model_state[k] = deepcopy(v)

        # Save optimizer state
        self.optimizer_state = deepcopy(optimizer.state_dict())

        self.best_lr = None
        self.history = {"lr": [], "loss": []}

    def find(
        self,
        dataloader,
        start_lr=1e-7,
        end_lr=10,
        num_iter=100,
        step_mode="exp",
        diverge_th=5.0,
        smoothing_factor=0.05,
    ):
        """Run the learning rate finder.

        Args:
            dataloader: DataLoader for training data
            start_lr: Starting learning rate
            end_lr: Maximum learning rate to try
            num_iter: Number of iterations to run
            step_mode: How to step the learning rate ('exp' or 'linear')
            diverge_th: Threshold for divergence (multiplier on lowest loss)
            smoothing_factor: Exponential moving average smoothing factor

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
        self.history = {"lr": [], "loss": [], "raw_loss": []}
        iteration = 0
        data_iter = iter(dataloader)

        # For tracking minimum loss
        min_loss = float("inf")
        smoothed_loss = None

        while iteration < num_iter:
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                inputs, targets = next(data_iter)

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Add exception handling for model forward pass
            try:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Check for NaN or inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(
                        f"Training diverged at learning rate: {self.optimizer.param_groups[0]['lr']:.10f}"
                    )
                    break

                # Update minimum loss
                min_loss = min(min_loss, loss.item())

                # Apply smoothing to the loss
                if smoothed_loss is None:
                    smoothed_loss = loss.item()
                else:
                    smoothed_loss = (
                        smoothed_loss * (1 - smoothing_factor)
                        + loss.item() * smoothing_factor
                    )

                # Record the learning rate and both raw and smoothed loss
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.history["lr"].append(current_lr)
                self.history["loss"].append(smoothed_loss)
                self.history["raw_loss"].append(loss.item())

                # Check for divergence (stop if loss grows too much)
                if loss.item() > diverge_th * min_loss and iteration > 10:
                    print(
                        f"Stopping early at LR {current_lr:.10f}: loss {loss.item():.6f} > {diverge_th} × minimum loss {min_loss:.6f}"
                    )
                    break

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Update learning rate for next iteration
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr_schedule(iteration)

                # Print progress occasionally
                if iteration % 100 == 0:
                    print(
                        f"Iteration {iteration}: lr={current_lr:.10f}, loss={loss.item():.6f}"
                    )

            except RuntimeError as e:
                print(
                    f"Error at learning rate {self.optimizer.param_groups[0]['lr']:.10f}: {str(e)}"
                )
                break

            iteration += 1

        # Restore the model and optimizer states
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

        # Only analyze if we have enough data points
        if len(self.history["loss"]) > 20:
            self._analyze_loss_curve()
        else:
            print(
                "Not enough data points collected to suggest an optimal learning rate."
            )

        return self.history["lr"], self.history["loss"]

    def _analyze_loss_curve(self):
        """Analyze the loss curve to find the optimal learning rate using multiple methods."""
        lrs = np.array(self.history["lr"])
        losses = np.array(self.history["loss"])

        # 1. Apply Savitzky-Golay filter for better smoothing with large number of points
        window_length = min(51, len(losses) // 10 * 2 + 1)  # Must be odd
        if window_length >= 5:  # Need at least 5 points for savgol_filter
            try:
                smooth_losses = savgol_filter(losses, window_length, 3)
            except Exception as e:
                print(f"Warning: Could not apply Savitzky-Golay filter: {e}")
                smooth_losses = losses
        else:
            smooth_losses = losses

        # 2. Calculate gradients of the smoothed loss curve
        try:
            gradients = np.gradient(smooth_losses)

            # 3. Find the region of consistent decrease (not just the steepest point)
            # This helps avoid picking points in noisy early regions

            # First, identify where significant decrease begins (loss drops by at least 10%)
            initial_loss = smooth_losses[0]
            significant_decrease_idx = np.where(smooth_losses < 0.9 * initial_loss)[0]

            if len(significant_decrease_idx) > 0:
                decrease_start_idx = significant_decrease_idx[0]
            else:
                decrease_start_idx = 0

            # Find the minimum point
            min_loss_idx = np.argmin(smooth_losses)

            # Look for steep descent in the region between start of decrease and minimum
            valid_region = slice(decrease_start_idx, min_loss_idx)

            if (
                valid_region.start < valid_region.stop
                and len(gradients[valid_region]) > 0
            ):
                # Find the point with steepest descent in valid region
                steepest_idx = valid_region.start + np.argmin(gradients[valid_region])

                # Calculate the midpoint between start of decrease and minimum
                # This is often a good balance point in the steep section
                midpoint_idx = (
                    decrease_start_idx + (min_loss_idx - decrease_start_idx) // 2
                )

                # Calculate loss decrease percentage from start to minimum
                if initial_loss > 0:
                    decrease_percent = (
                        (initial_loss - smooth_losses[min_loss_idx])
                        / initial_loss
                        * 100
                    )
                    print(
                        f"Loss decreases by {decrease_percent:.1f}% from start to minimum"
                    )

                # If we have a significant drop (>10%), prefer a point in the steepest part of the curve
                if decrease_percent > 10:
                    # Choose a point 1/3 of the way from steepest point to minimum
                    # This places us in steep descent but with a bias toward the minimum
                    best_idx = steepest_idx + (min_loss_idx - steepest_idx) // 3
                    self.best_lr = lrs[best_idx]
                    print(
                        f"Significant loss drop detected. Suggesting LR from steep descent region: {self.best_lr:.6e}"
                    )
                else:
                    # For small drops, be more conservative and use the steepest point
                    self.best_lr = lrs[steepest_idx]
                    print(
                        f"Small loss drop detected. Suggesting LR from steepest point: {self.best_lr:.6e}"
                    )

                # Also calculate alternative suggestions for user reference
                self.steepest_lr = lrs[steepest_idx]
                self.min_loss_lr = lrs[min_loss_idx]
                self.midpoint_lr = lrs[midpoint_idx]

                print(f"Alternative learning rates:")
                print(f"  - Steepest descent: {self.steepest_lr:.6e}")
                print(f"  - Middle of descent: {self.midpoint_lr:.6e}")
                print(f"  - Minimum loss: {self.min_loss_lr:.6e}")
            else:
                # Fallback to traditional method if region analysis fails
                best_idx = np.argmin(gradients)
                self.best_lr = lrs[best_idx]
                print(
                    f"Using traditional steepest gradient approach: {self.best_lr:.6e}"
                )
        except Exception as e:
            print(f"Error analyzing loss curve: {e}")
            if len(losses) > 0:
                # Very simple fallback - just use the LR at minimum loss
                min_idx = np.argmin(losses)
                self.best_lr = lrs[min_idx]
                print(f"Fallback to LR at minimum loss: {self.best_lr:.6e}")

    def plot(self, log_scale=True, suggest_lr=True, skip_start=5, skip_end=5):
        """Plot the learning rate vs. loss curve with comprehensive analysis.

        Args:
            log_scale (bool): Whether to use log scale for x-axis
            suggest_lr (bool): Whether to indicate the suggested LR
            skip_start (int): Number of points to skip at the start
            skip_end (int): Number of points to skip at the end

        Returns:
            matplotlib.figure.Figure: The generated plot
        """
        plt.figure(figsize=(12, 8))

        # Skip some initial and final points for better visualization
        if len(self.history["lr"]) > skip_start + skip_end:
            lrs = (
                self.history["lr"][skip_start:-skip_end]
                if skip_end > 0
                else self.history["lr"][skip_start:]
            )
            losses = (
                self.history["loss"][skip_start:-skip_end]
                if skip_end > 0
                else self.history["loss"][skip_start:]
            )
        else:
            lrs = self.history["lr"]
            losses = self.history["loss"]

        plt.plot(lrs, losses, "-", color="blue", alpha=0.7, label="Smoothed Loss")

        # Plot raw loss points for reference if available
        if "raw_loss" in self.history and len(self.history["raw_loss"]) > 0:
            raw_losses = (
                self.history["raw_loss"][skip_start:-skip_end]
                if skip_end > 0
                else self.history["raw_loss"][skip_start:]
            )
            if len(raw_losses) == len(lrs):
                plt.plot(
                    lrs,
                    raw_losses,
                    ".",
                    color="lightblue",
                    alpha=0.4,
                    markersize=2,
                    label="Raw Loss",
                )

        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True, linestyle="--", alpha=0.6)

        if log_scale:
            plt.xscale("log")

        # Mark the minimum loss point
        min_loss_idx = np.argmin(losses) if len(losses) > 0 else -1
        if min_loss_idx >= 0:
            plt.scatter(
                lrs[min_loss_idx],
                losses[min_loss_idx],
                color="green",
                s=75,
                label=f"Min Loss at LR={lrs[min_loss_idx]:.2e}",
            )

        # Mark all suggested learning rates
        if suggest_lr:
            if hasattr(self, "best_lr") and self.best_lr:
                plt.axvline(
                    x=self.best_lr,
                    color="r",
                    linestyle="--",
                    label=f"Suggested LR: {self.best_lr:.2e}",
                )

            # Add other reference points if available
            if hasattr(self, "steepest_lr") and self.steepest_lr:
                plt.axvline(
                    x=self.steepest_lr,
                    color="orange",
                    linestyle=":",
                    alpha=0.7,
                    label=f"Steepest Descent: {self.steepest_lr:.2e}",
                )

            if hasattr(self, "midpoint_lr") and self.midpoint_lr:
                plt.axvline(
                    x=self.midpoint_lr,
                    color="purple",
                    linestyle="-.",
                    alpha=0.7,
                    label=f"Mid-Descent: {self.midpoint_lr:.2e}",
                )

        plt.legend(loc="best")
        return plt.gcf()
