"""Tests for the validator modules."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.training.validators.validator import validate, validate_size_mismatch_fix
from src.losses.loss_factory import create_loss


def test_validator(device):
    """Test the validate function."""

    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = SimpleModel().to(device)

    # Create a simple dataset
    x = torch.ones(10, 3, 1, 8, 8).to(device)
    y = torch.ones(10, 3, 1, 8, 8).to(device)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2)

    # Create loss function
    criterion = create_loss({"loss_params": {"type": "mse"}})

    # Run validation
    loss = validate(model, dataloader, criterion, device)

    # Since model returns x and y == x, loss should be 0
    assert loss == 0.0, f"Expected loss 0.0, got {loss}"


def test_validator_with_error(device):
    """Test the validate function with a model that produces errors."""

    # Create a model that outputs 2*input
    class ErrorModel(torch.nn.Module):
        def forward(self, x):
            return 2 * x

    model = ErrorModel().to(device)

    # Create a simple dataset
    x = torch.ones(10, 3, 1, 8, 8).to(device)
    y = torch.ones(10, 3, 1, 8, 8).to(device)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2)

    # Create loss function
    criterion = create_loss({"loss_params": {"type": "mse"}})

    # Run validation
    loss = validate(model, dataloader, criterion, device)

    # Since model returns 2*x and y == x, MSE should be (2-1)² = 1.0
    assert abs(loss - 1.0) < 1e-5, f"Expected loss 1.0, got {loss}"


def test_validator_size_mismatch_fix(device):
    """Test the validator with size mismatch fix."""

    # Create a model that returns a tensor with t=1
    class ShrinkTimeModel(torch.nn.Module):
        def forward(self, x):
            # Return only the last time step, reducing time dimension to 1
            return x[:, :, -1:, :, :]

    model = ShrinkTimeModel().to(device)

    # Create dataset with longer time dimension in target
    x = torch.ones(10, 3, 5, 8, 8).to(device)  # t=5
    y = torch.ones(10, 3, 5, 8, 8).to(device)  # t=5
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2)

    # Create loss function
    criterion = create_loss({"loss_params": {"type": "mse"}})

    # Run validation with size mismatch fix
    loss = validate_size_mismatch_fix(model, dataloader, criterion, device)

    # Since we're comparing only the last time step and values are the same,
    # loss should be close to 0
    assert loss < 1e-5, f"Expected loss close to 0, got {loss}"
