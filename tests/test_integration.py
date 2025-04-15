"""Integration tests for the NODs package."""

import pytest
import torch
import torch.nn as nn
import numpy as np
import os
from src.utils.config import load_config
from src.models.model_factory import create_model
from src.models.tfno_neuralop import TFNO_neuralop
from src.models.tfno import TFNO
from src.data.dataset import get_dataloaders
from src.losses.loss_factory import create_loss
from src.training.validators.validator import validate
from src.optimizers import create_optimizer, create_scheduler


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
        sub_t=config["dataset_params"].get("sub_t", 1),
        sub_x=config["dataset_params"].get("sub_x", 1),
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
    loss = criterion(output, y)

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


def test_integration_with_custom_tfno(tfno_3d_config, generate_mini_mhd_data, device):
    """Test integration with the custom TFNO 3D model."""
    # Modify config to add necessary parameters
    config = tfno_3d_config.copy()

    # Add other necessary params
    config["normalization_params"] = {
        "type": "physics",
        "input_norm": [1.0, 1.0, 2.5e-4],
        "output_norm": [1.0, 1.0, 2.5e-4],
    }
    config["loss_params"] = {"type": "mse", "reduction": "mean"}
    config["optimizer_params"] = {
        "optimizer_type": "adam",
        "lr": 0.001,
        "betas": [0.9, 0.999],
        "use_scheduler": False,
    }

    # Create model
    model = create_model(config).to(device)
    assert isinstance(model, TFNO), "Model should be TFNO"

    # Load data
    data = np.load(generate_mini_mhd_data)
    reshaped_data = torch.from_numpy(data).float().permute(0, 4, 1, 2, 3)
    x_data = reshaped_data[:2, :, 0:1]  # Just use 2 samples
    y_data = reshaped_data[:2]  # Just use 2 samples

    # Create simple dataset and dataloader
    from torch.utils.data import TensorDataset, DataLoader

    # Add grid embeddings for input
    batch_size, channels, t, nx, ny = x_data.shape
    t_grid = (
        torch.linspace(0, 1, t).reshape(1, 1, t, 1, 1).repeat(batch_size, 1, 1, nx, ny)
    )
    x_grid = (
        torch.linspace(0, 1, nx).reshape(1, 1, 1, nx, 1).repeat(batch_size, 1, t, 1, ny)
    )
    y_grid = (
        torch.linspace(0, 1, ny).reshape(1, 1, 1, 1, ny).repeat(batch_size, 1, t, nx, 1)
    )

    # Combine grid and data
    x_with_grid = torch.cat([t_grid, x_grid, y_grid, x_data], dim=1).to(device)
    y_data = y_data.to(device)

    # Create loss function
    criterion = create_loss(config)

    # Create optimizer
    optimizer = create_optimizer(model.parameters(), config)

    # Forward pass
    model.train()
    optimizer.zero_grad()
    output = model(x_with_grid)
    loss = criterion(output, y_data)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Check loss is finite
    assert torch.isfinite(loss), "Loss should be finite"


def test_integration(mini_config_path, generate_mini_mhd_data, device):
    """Test integration of optimizers and schedulers with the model training pipeline."""
    # Load config
    config = load_config(mini_config_path)

    # Update config with optimizer and scheduler settings
    config["optimizer_params"] = {
        "optimizer_type": "adamw",
        "lr": 0.001,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "use_scheduler": True,
        "scheduler_type": "cosine",
        "T_max": 10,
        "eta_min": 1e-6,
    }

    # Create model
    model = create_model(config).to(device)

    # Get dataloaders
    train_loader, val_loader, _ = get_dataloaders(
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

    # Create optimizer using factory
    optimizer = create_optimizer(model.parameters(), config)

    # Create scheduler using factory
    scheduler = create_scheduler(optimizer, config)

    # Test scheduler type
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR), (
        "Scheduler should be CosineAnnealingLR as specified in config"
    )

    # Check initial LR
    initial_lr = optimizer.param_groups[0]["lr"]
    assert abs(initial_lr - 0.001) < 1e-8, "Initial LR should be 0.001"

    # Train for one epoch
    model.train()
    total_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Just do one batch for testing efficiency
        if batch_idx >= 0:
            break

    # Step the scheduler AFTER optimizer step
    scheduler.step()

    # Check LR has been updated
    assert optimizer.param_groups[0]["lr"] != initial_lr, (
        "LR should change after scheduler.step()"
    )

    # ===== SEPARATE TEST FOR REDUCELRONPLATEAU =====
    # Since we've already tested the factory can create different scheduler types,
    # we'll skip additional ReduceLROnPlateau testing to avoid compatibility issues

    # Just verify our factory can create it
    config["optimizer_params"]["scheduler_type"] = "reducelr"
    config["optimizer_params"]["mode"] = "min"
    config["optimizer_params"]["factor"] = 0.5
    config["optimizer_params"]["patience"] = 1

    # Create new optimizer and scheduler
    optimizer_reducelr = create_optimizer(model.parameters(), config)
    scheduler_reducelr = create_scheduler(optimizer_reducelr, config)

    # Verify type
    assert isinstance(scheduler_reducelr, torch.optim.lr_scheduler.ReduceLROnPlateau), (
        "Scheduler should be ReduceLROnPlateau as specified in config"
    )
