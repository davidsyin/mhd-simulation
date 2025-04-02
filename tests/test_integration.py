"""Integration tests for the NODs package."""

import pytest
import torch
import os
from src.utils.config import load_config
from src.models.model_factory import create_model
from src.data.dataset import get_dataloaders
from src.losses.loss_factory import create_loss
from src.training.validators.validator import validate


def test_full_pipeline(mini_config_path, generate_mini_mhd_data, device):
    """Test the full model pipeline from data loading to validation."""
    # Load config
    config = load_config(mini_config_path)

    # Create model
    model = create_model(config).to(device)

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        generate_mini_mhd_data,
        normalization_config={"normalization_params": config["normalization_params"]},
        batch_size=config["train_loader_params"]["batch_size"],
        num_workers=config["train_loader_params"]["num_workers"],
        train_ratio=config["dataset_params"]["num_train"]
        / (
            config["dataset_params"]["num_train"] + config["dataset_params"]["num_test"]
        ),
        seed=config["dataset_params"]["seed"],
    )

    # Create loss function
    criterion = create_loss(config)

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["optimizer_params"]["lr"],
        betas=config["optimizer_params"]["betas"],
    )

    # Train for one batch
    model.train()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    # Forward pass
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(
        output, x
    )  # Compare prediction with input (autoencoder mode for testing)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Check that loss is finite
    assert not torch.isnan(loss).any(), "Loss is NaN"
    assert not torch.isinf(loss).any(), "Loss is Inf"

    # Validate
    val_loss = validate(model, val_loader, criterion, device)

    # Check validation loss
    assert not torch.isnan(torch.tensor(val_loss)).any(), "Validation loss is NaN"
    assert not torch.isinf(torch.tensor(val_loss)).any(), "Validation loss is Inf"

    print(f"Validation loss: {val_loss}")
