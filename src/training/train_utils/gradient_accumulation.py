"""
Gradient accumulation utilities for training with larger effective batch sizes.

This module provides utilities to perform gradient accumulation, which allows
training with larger effective batch sizes than would fit in GPU memory by
accumulating gradients over multiple forward and backward passes.
"""

import torch


class GradientAccumulator:
    """Utility for gradient accumulation during training.

    Args:
        model: The neural network model
        optimizer: The optimizer used for training
        accumulation_steps (int): Number of steps to accumulate gradients
    """

    def __init__(self, model, optimizer, accumulation_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def __call__(self, inputs, targets, criterion):
        """Perform forward and backward passes with gradient accumulation.

        Args:
            inputs: Input data
            targets: Target data
            criterion: Loss function

        Returns:
            float: Current batch loss value
        """
        # Forward pass
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)

        # Scale the loss to account for accumulation
        scaled_loss = loss / self.accumulation_steps

        # Backward pass
        scaled_loss.backward()

        # Update weights if we've accumulated enough gradients
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.current_step = 0

        return loss.item()

    def zero_grad(self):
        """Zero out the gradients."""
        self.optimizer.zero_grad()
        self.current_step = 0


# Example usage in a training loop:
"""
# Setup
model = create_model(config)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
accumulator = GradientAccumulator(model, optimizer, accumulation_steps=4)

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # This handles forward, backward, and occasional optimizer steps
        loss = accumulator(inputs, targets, criterion)
        
    # Make sure to step at the end of an epoch if needed
    if accumulator.current_step > 0:
        optimizer.step()
        optimizer.zero_grad()
"""
