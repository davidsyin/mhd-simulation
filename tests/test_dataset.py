"""Tests for the dataset module."""

import pytest
import torch
import numpy as np
from src.data.dataset import MHDDataset, get_dataloaders
from src.utils.config import load_config


def test_mhd_dataset(generate_mini_mhd_data, mini_config):
    """Test MHDDataset initialization and basic functionality."""
    # Load the data before passing to MHDDataset
    data_path = generate_mini_mhd_data
    data = np.load(data_path)

    # Create MHDDataset for train split with loaded data
    train_dataset = MHDDataset(
        data,
        normalization_config={
            "normalization_params": mini_config["normalization_params"]
        },
        split="train",
        train_ratio=0.8,
        seed=42,
    )

    # Create MHDDataset for val split with loaded data
    val_dataset = MHDDataset(
        data,
        normalization_config={
            "normalization_params": mini_config["normalization_params"]
        },
        split="val",
        train_ratio=0.8,
        seed=42,
    )

    # Create MHDDataset for test split with loaded data
    test_dataset = MHDDataset(
        data,
        normalization_config={
            "normalization_params": mini_config["normalization_params"]
        },
        split="test",
        train_ratio=0.8,
        seed=42,
    )

    # Test dataset lengths
    assert len(train_dataset) == 8, "Train dataset should have 8 samples"
    assert len(val_dataset) == 1, "Train dataset should have 1 samples"
    assert len(test_dataset) == 1, "Test dataset should have 1 samples"

    # Test shape of items
    x, y = train_dataset[0]
    assert x.shape == (3, 1, 32, 32), "Input tensor should have shape (3, 1, 32, 32)"
    assert y.shape == (3, 21, 32, 32), "Target tensor should have shape (3, 21, 32, 32)"

    # Check normalization
    assert train_dataset.normalizer is not None, "Normalizer should be initialized"
    assert hasattr(
        train_dataset.normalizer, "normalize"
    ), "Normalizer should have normalize method"
    assert hasattr(
        train_dataset.normalizer, "denormalize"
    ), "Normalizer should have denormalize method"


def test_get_dataloaders(generate_mini_mhd_data, mini_config):
    """Test get_dataloaders function."""
    data_path = generate_mini_mhd_data

    # Create dataloaders - this function handles loading the data internally
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path,
        normalization_config={
            "normalization_params": mini_config["normalization_params"]
        },
        batch_size=2,
        num_workers=0,
        train_ratio=0.8,
        seed=42,
    )

    # Test dataloader lengths
    assert (
        len(train_loader) == 4
    ), "Train loader should have 4 batches (8 samples with batch size 2)"
    assert (
        len(val_loader) == 1
    ), "Val loader should have 1 batch (1 sample with batch size 2)"
    assert (
        len(test_loader) == 1
    ), "Test loader should have 1 batch (1 sample with batch size 2)"

    # Test shape of batches
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape == (
        2,
        3,
        1,
        32,
        32,
    ), "Input batch should have shape (2, 3, 1, 32, 32)"
    assert y_batch.shape == (
        2,
        3,
        21,
        32,
        32,
    ), "Target batch should have shape (2, 3, 21, 32, 32)"


def test_dataset_with_config(generate_mini_mhd_data, mini_config_path):
    """Test loading dataloaders using the config file."""
    config = load_config(mini_config_path)

    # Create dataloaders using config - get_dataloaders handles the data loading
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

    # Check that the dataloaders are not empty
    assert len(train_loader) > 0, "Train loader should not be empty"
    assert len(val_loader) > 0, "Val loader should not be empty"
    assert len(test_loader) > 0, "Test loader should not be empty"
