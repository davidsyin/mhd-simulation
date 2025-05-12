"""Tests for the validator modules."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.training.validators import validate
from src.losses.loss_factory import create_loss


def test_validator(device):
    """Test the validate function."""

    # Create a simple model that outputs same shape as targets
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            # Return a tensor with same shape as target (removing grid coords)
            return x[:, 3:, :, :, :]  # Skip the first 3 channels (grid embeddings)

    model = SimpleModel().to(device)

    # Create a simple dataset with grid embeddings
    # Input x has 6 channels (3 grid + 3 data), output y has 3 channels
    x = torch.ones(10, 6, 1, 8, 8).to(device)  # 6 channels
    y = torch.ones(10, 3, 1, 8, 8).to(device)  # 3 channels
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2)

    # Create loss function
    criterion = create_loss({"loss_params": {"type": "mse"}})

    # Run validation
    loss = validate(model, dataloader, criterion, device)

    # Since model returns data part of x and y == data part of x, loss should be 0
    assert loss < 1e-5, f"Expected loss close to 0.0, got {loss}"


def test_validator_with_error(device):
    """Test the validate function with a model that produces errors."""

    # Create a model that outputs 2 * data part of input
    class ErrorModel(torch.nn.Module):
        def forward(self, x):
            # Double just the data channels
            return 2 * x[:, 3:, :, :, :]  # Skip the grid embedding channels

    model = ErrorModel().to(device)

    # Create a simple dataset with grid embeddings
    x = torch.ones(10, 6, 1, 8, 8).to(device)  # 6 channels
    y = torch.ones(10, 3, 1, 8, 8).to(device)  # 3 channels
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2)

    # Create loss function
    criterion = create_loss({"loss_params": {"type": "mse"}})

    # Run validation
    loss = validate(model, dataloader, criterion, device)

    # Since model returns 2*data and y == data, MSE should be (2-1)² = 1.0
    assert abs(loss - 1.0) < 1e-5, f"Expected loss 1.0, got {loss}"
