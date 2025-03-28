"""
Automatic Mixed Precision (AMP) training utilities.

This module provides utilities for mixed precision training using PyTorch's
native AMP support, which can significantly speed up training on modern GPUs.
"""

import torch
from torch.cuda.amp import autocast, GradScaler


class AMPTrainer:
    """Utility for automatic mixed precision training.

    Args:
        model: The neural network model
        optimizer: The optimizer used for training
        scaler_kwargs: Additional arguments for GradScaler
    """

    def __init__(self, model, optimizer, **scaler_kwargs):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler(**scaler_kwargs)

    def train_step(self, inputs, targets, criterion):
        """Perform a single training step with mixed precision.

        Args:
            inputs: Input data
            targets: Target data
            criterion: Loss function

        Returns:
            float: Current batch loss value
        """
        # Forward pass with autocast
        self.optimizer.zero_grad()
        with autocast():
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

        # Backward pass with scaled gradients
        self.scaler.scale(loss).backward()

        # Unscale gradients and clip if needed
        # self.scaler.unscale_(self.optimizer)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step with scaled gradients
        self.scaler.step(self.optimizer)

        # Update scaler
        self.scaler.update()

        return loss.item(), outputs

    def save_state(self, checkpoint_dict):
        """Save AMP state in checkpoint.

        Args:
            checkpoint_dict: Dictionary containing checkpoint data

        Returns:
            dict: Updated checkpoint dictionary with AMP state
        """
        checkpoint_dict["amp_scaler"] = self.scaler.state_dict()
        return checkpoint_dict

    def load_state(self, checkpoint_dict):
        """Load AMP state from checkpoint.

        Args:
            checkpoint_dict: Dictionary containing checkpoint data
        """
        if "amp_scaler" in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict["amp_scaler"])


# Example usage in a training loop:
"""
# Setup
model = create_model(config)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
amp_trainer = AMPTrainer(model, optimizer)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward and backward pass with mixed precision
        loss, outputs = amp_trainer.train_step(inputs, targets, criterion)
        
# Save checkpoint with AMP state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}
checkpoint = amp_trainer.save_state(checkpoint)
torch.save(checkpoint, 'checkpoint.pt')
"""
