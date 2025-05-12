"""Tests for the checkpoint functionality."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import shutil
from datetime import datetime

from src.utils.config import load_config
from src.neurops import create_model
from src.data.dataset import MHDDataset
from src.training.train_utils.checkpoints import save_checkpoint, load_checkpoint


@pytest.fixture
def test_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "test_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    yield checkpoint_dir


@pytest.fixture
def random_mhd_data():
    """Create random MHD simulation data for testing."""
    num_samples = 12
    timesteps = 10
    nx = 16
    ny = 16
    channels = 3

    data = np.random.randn(num_samples, timesteps, nx, ny, channels).astype(np.float32)
    return torch.from_numpy(data).float()


def test_save_load_checkpoint(mini_config, test_checkpoint_dir, device):
    """Test saving and loading a model checkpoint."""
    model = create_model(mini_config).to(device)

    optimizer_params = mini_config.get("optimizer_params", {})
    lr = optimizer_params.get("lr", 1e-4)
    weight_decay = optimizer_params.get("weight_decay", 1e-5)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if optimizer_params.get("use_scheduler", False):
        scheduler_type = optimizer_params.get("scheduler_type", "reducelr")
        if scheduler_type == "reducelr":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=optimizer_params.get("mode", "min"),
                factor=optimizer_params.get("factor", 0.7),
                patience=optimizer_params.get("patience", 2),
            )

    checkpoint_path = os.path.join(test_checkpoint_dir, "model_checkpoint.pt")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=5,
        loss=0.1,
        scheduler=scheduler,
        ckpt_path=checkpoint_path,
        denorm_loss_rel_l2=0.5,
        denorm_loss_mse=0.5,
    )

    assert os.path.exists(checkpoint_path), "Checkpoint file should exist"

    new_model = create_model(mini_config).to(device)
    new_optimizer = optim.AdamW(
        new_model.parameters(), lr=lr, weight_decay=weight_decay
    )

    new_scheduler = None
    if optimizer_params.get("use_scheduler", False):
        if scheduler_type == "reducelr":
            new_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                new_optimizer,
                mode=optimizer_params.get("mode", "min"),
                factor=optimizer_params.get("factor", 0.7),
                patience=optimizer_params.get("patience", 2),
            )

    checkpoint_data = load_checkpoint(
        ckpt_path=checkpoint_path,
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
    )

    assert checkpoint_data["epoch"] == 5, "Loaded epoch should be 5"
    assert checkpoint_data["loss"] == pytest.approx(0.1), "Loaded loss should be 0.1"
    assert checkpoint_data["denorm_loss_rel_l2"] == pytest.approx(0.5), (
        "Loaded denorm_loss_rel_l2 should be 0.5"
    )
    assert checkpoint_data["denorm_loss_mse"] == pytest.approx(0.5), (
        "Loaded denorm_loss_mse should be 0.5"
    )

    original_params = list(model.parameters())
    loaded_params = list(new_model.parameters())

    for p1, p2 in zip(original_params, loaded_params):
        assert torch.allclose(p1, p2), "Model parameters should match after loading"


def test_checkpoint_with_dataset(
    mini_config, random_mhd_data, test_checkpoint_dir, device
):
    """Test saving and loading checkpoints with MHDDataset."""
    # Reshape the data to match expected format for MHDDataset
    # From (num_samples, timesteps, nx, ny, channels) to (num_samples, channels, timesteps, nx, ny)
    data_tensor = random_mhd_data.permute(0, 4, 1, 2, 3)

    # Extract initial conditions: first timestep only
    x_data = data_tensor[:, :, 0:1, :, :]
    y_data = data_tensor

    # Create the dataset
    # Simplified normalization config
    norm_config = {
        "normalization_params": {
            "type": "physics",
            "input_norm": [1.0, 1.0, 1.0],
            "output_norm": [1.0, 1.0, 1.0],
        }
    }

    train_dataset = MHDDataset(
        x_data=x_data,
        y_data=y_data,
        normalization_config=norm_config,
        split="train",
        train_ratio=0.8,
        seed=42,
        sub_t=1,
        sub_x=1,
    )

    batch_size = 2
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    model = create_model(mini_config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=2
    )

    checkpoint_path = os.path.join(test_checkpoint_dir, "model_checkpoint_dataset.pt")

    criterion = nn.MSELoss()

    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batch_count += 1

    avg_loss = epoch_loss / batch_count

    scheduler.step(avg_loss)

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        loss=avg_loss,
        scheduler=scheduler,
        ckpt_path=checkpoint_path,
        denorm_loss_rel_l2=avg_loss * 10,
        denorm_loss_mse=avg_loss * 10,  # Just a dummy denormalized loss for testing
    )

    # Create a new model instance to test loading
    new_model = create_model(mini_config).to(device)
    new_optimizer = optim.AdamW(new_model.parameters(), lr=1e-4, weight_decay=1e-5)
    new_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        new_optimizer, mode="min", factor=0.7, patience=2
    )

    checkpoint_data = load_checkpoint(
        ckpt_path=checkpoint_path,
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
    )

    # Verify models have the same weights after loading
    original_params = list(model.parameters())
    loaded_params = list(new_model.parameters())

    models_match = all(
        torch.allclose(p1, p2) for p1, p2 in zip(original_params, loaded_params)
    )

    assert models_match, "Model weights should match after loading"

    # Test forward pass with loaded model
    with torch.no_grad():
        test_inputs, test_targets = next(iter(train_loader))
        test_inputs = test_inputs.to(device)

        original_output = model(test_inputs)
        loaded_output = new_model(test_inputs)

        assert torch.allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5), (
            "Model outputs should match after loading"
        )


def test_checkpoint_nonexistent_path(mini_config, device):
    """Test loading a checkpoint from a non-existent path."""
    model = create_model(mini_config).to(device)

    # Create a non-existent path
    non_existent_path = "non_existent_checkpoint.pt"

    # Load checkpoint from non-existent path
    fallback_data = load_checkpoint(ckpt_path=non_existent_path, model=model)

    # Verify fallback values
    assert fallback_data["epoch"] == 0, "Fallback epoch should be 0"
    assert fallback_data["loss"] == float("inf"), "Fallback loss should be infinity"
    assert fallback_data["denorm_loss_rel_l2"] == float("inf"), (
        "Fallback denorm_loss_rel_l2 should be infinity"
    )
    assert fallback_data["denorm_loss_mse"] == float("inf"), (
        "Fallback denorm_loss_mse should be infinity"
    )


def test_checkpoint_distributed_data_parallel(mini_config, test_checkpoint_dir, device):
    """Test saving and loading a checkpoint with DistributedDataParallel model."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping DDP test")

    model = create_model(mini_config).to(device)

    # Wrap model in DDP (without initializing process group - just for testing)
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[0])
    except:
        pytest.skip("Failed to initialize DDP, skipping test")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    checkpoint_path = os.path.join(test_checkpoint_dir, "model_checkpoint_ddp.pt")

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=3,
        loss=0.2,
        scheduler=None,
        ckpt_path=checkpoint_path,
    )

    new_model = create_model(mini_config).to(device)
    new_model = DDP(new_model, device_ids=[0])
    new_optimizer = optim.AdamW(new_model.parameters(), lr=1e-4)

    checkpoint_data = load_checkpoint(
        ckpt_path=checkpoint_path, model=new_model, optimizer=new_optimizer
    )

    # Verify loaded data
    assert checkpoint_data["epoch"] == 3, "Loaded epoch should be 3"
    assert checkpoint_data["loss"] == pytest.approx(0.2), "Loaded loss should be 0.2"

    # Verify models have the same weights
    original_params = list(model.module.parameters())
    loaded_params = list(new_model.module.parameters())

    for p1, p2 in zip(original_params, loaded_params):
        assert torch.allclose(p1, p2), "Model parameters should match after loading"
