"""Tests for the diffusion dataset module."""

import pytest
import torch
import numpy as np
import os
import tempfile
from torch.utils.data import DataLoader

from src.data import DiffusionDataset, get_diffusion_dataloaders
from src.normalizations import create_normalization


@pytest.fixture
def mock_diffusion_data():
    """Create mock diffusion dataset files for testing."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample data dimensions
        n_samples = 5
        channels = 3
        timesteps = 10
        nx, ny = 16, 16
        
        # Create sample input and target data
        inputs = np.random.randn(n_samples, channels, timesteps, nx, ny).astype(np.float32)
        targets = np.random.randn(n_samples, channels, timesteps, nx, ny).astype(np.float32)
        
        # Create data dictionaries
        train_data = {"diff_inputs": inputs[:3], "diff_targets": targets[:3]}
        val_data = {"diff_inputs": inputs[3:4], "diff_targets": targets[3:4]}
        test_data = {"diff_inputs": inputs[4:], "diff_targets": targets[4:]}
        
        # Save to npz files
        train_path = os.path.join(temp_dir, "train_diffusion.npz")
        val_path = os.path.join(temp_dir, "val_diffusion.npz")
        test_path = os.path.join(temp_dir, "test_diffusion.npz")
        
        np.savez(train_path, **train_data)
        np.savez(val_path, **val_data)
        np.savez(test_path, **test_data)
        
        # Create statistics file for normalization
        mean = np.array([0.0, 0.0, 0.0])
        std = np.array([1.0, 1.0, 1.0])
        
        stats_path = os.path.join(temp_dir, "diffusion_stats.npz")
        np.savez(stats_path, mean=mean, std=std)
        
        # Return paths
        yield {
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
            "stats_path": stats_path,
            "data_shape": (n_samples, channels, timesteps, nx, ny)
        }


@pytest.fixture
def diffusion_config(mock_diffusion_data):
    """Create a configuration for diffusion testing."""
    return {
        "model_params": {
            "model_type": "elucidated",
            "channels": 3,  # Expected output channels
            "image_size": 16,
            "num_sample_steps": 20,
            "sigma_min": 0.002,
            "sigma_max": 80,
            "sigma_data": 0.5,  # Will be overridden by dataset
        },
        "normalization_params": {
            "type": "standard",
            "inputs_stats_file": mock_diffusion_data["stats_path"],
            "targets_stats_file": mock_diffusion_data["stats_path"]
        },
        "dataset_params": {
            "train_path": mock_diffusion_data["train_path"],
            "val_path": mock_diffusion_data["val_path"],
            "test_path": mock_diffusion_data["test_path"],
            # Optional: select specific channels
            "channel_indices": [0, 1, 2]
        },
        "train_loader_params": {
            "batch_size": 2,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": True
        }
    }


@pytest.fixture
def standard_normalizer():
    """Create a standard normalizer for testing."""
    return create_normalization({
        "normalization_params": {
            "type": "standard",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0]
        }
    })


def test_diffusion_dataset_init(mock_diffusion_data):
    """Test DiffusionDataset initialization."""
    # Create a basic dataset without normalization
    dataset = DiffusionDataset(
        mock_diffusion_data["train_path"],
        split_name="train"
    )
    
    # Check basic properties
    assert len(dataset) > 0
    assert hasattr(dataset, "inputs")
    assert hasattr(dataset, "targets")
    assert hasattr(dataset, "original_shape")
    assert hasattr(dataset, "timesteps")
    
    # Original shape should match input shape
    n_samples, channels, timesteps, nx, ny = mock_diffusion_data["data_shape"]
    assert dataset.original_shape[:2] == (3, channels)  # We used 3 samples for train
    assert dataset.timesteps == timesteps
    
    # Test with a dataset using normalizer
    normalizer = create_normalization({
        "normalization_params": {
            "type": "standard",
            "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0]
        }
    })
    
    dataset_norm = DiffusionDataset(
        mock_diffusion_data["train_path"],
        input_normalizer=normalizer,
        target_normalizer=normalizer,
        split_name="train"
    )
    
    # Check the dataset has been normalized
    assert isinstance(dataset_norm.inputs, torch.Tensor)
    assert isinstance(dataset_norm.targets, torch.Tensor)


def test_diffusion_dataset_prepare_input(mock_diffusion_data, diffusion_config, standard_normalizer):
    """Test the prepare_diffusion_input method."""
    # Create dataset with config and normalizers to ensure inputs are tensors
    dataset = DiffusionDataset(
        mock_diffusion_data["train_path"],
        config=diffusion_config,
        input_normalizer=standard_normalizer,
        target_normalizer=standard_normalizer,
        split_name="train"
    )
    
    # Check reshaped data
    n_samples, channels, timesteps, nx, ny = mock_diffusion_data["data_shape"]
    expected_samples = 3 * timesteps  # 3 train samples * timesteps
    expected_channels = diffusion_config["model_params"]["channels"]
    
    # Check that inputs and targets are PyTorch tensors
    assert isinstance(dataset.inputs, torch.Tensor)
    assert isinstance(dataset.targets, torch.Tensor)
    
    # Shape should be (n_samples*timesteps, channels, nx, ny)
    assert dataset.inputs.shape == (expected_samples, expected_channels, nx, ny)
    assert dataset.targets.shape == (expected_samples, expected_channels, nx, ny)


def test_diffusion_dataset_channel_selection(mock_diffusion_data, standard_normalizer):
    """Test channel selection in diffusion dataset."""
    # Create config that selects only the first channel
    config = {
        "model_params": {
            "channels": 1  # Request 1 channel output
        },
        "dataset_params": {
            "channel_indices": 0  # Select only the first channel
        }
    }
    
    # Use normalizers to ensure inputs are tensors
    dataset = DiffusionDataset(
        mock_diffusion_data["train_path"],
        config=config,
        input_normalizer=standard_normalizer,
        target_normalizer=standard_normalizer,
        split_name="train"
    )
    
    # Shape should reflect only one channel
    n_samples, channels, timesteps, nx, ny = mock_diffusion_data["data_shape"]
    expected_samples = 3 * timesteps  # 3 train samples * timesteps
    
    assert dataset.inputs.shape[1] == 1  # Only one channel
    assert dataset.targets.shape[1] == 1  # Only one channel
    
    # Create config that selects a subset of channels
    config = {
        "model_params": {
            "channels": 2  # Request 2 channel output
        },
        "dataset_params": {
            "channel_indices": [0, 2]  # Select first and third channel
        }
    }
    
    dataset = DiffusionDataset(
        mock_diffusion_data["train_path"],
        config=config,
        input_normalizer=standard_normalizer,
        target_normalizer=standard_normalizer,
        split_name="train"
    )
    
    # Shape should reflect two channels
    assert dataset.inputs.shape[1] == 2  # Two channels
    assert dataset.targets.shape[1] == 2  # Two channels


def test_diffusion_dataset_getitem(mock_diffusion_data):
    """Test __getitem__ method of the diffusion dataset."""
    dataset = DiffusionDataset(mock_diffusion_data["train_path"])
    
    # Get first item
    x, y = dataset[0]
    
    # Check types
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    
    # Check shapes (should match the shape after reshaping)
    assert len(x.shape) >= 3  # At least (channels, nx, ny)
    assert x.shape == y.shape  # Input and target should have same shape


def test_diffusion_dataset_sigma_data(mock_diffusion_data, diffusion_config, standard_normalizer):
    """Test sigma_data calculation in the diffusion dataset."""
    # Copy config to avoid modifying the fixture
    config = diffusion_config.copy()
    config["model_params"] = diffusion_config["model_params"].copy()
    
    # Initial sigma_data
    initial_sigma = config["model_params"]["sigma_data"]
    
    # Create dataset with normalizers to ensure inputs are tensors
    dataset = DiffusionDataset(
        mock_diffusion_data["train_path"],
        config=config,
        input_normalizer=standard_normalizer,
        target_normalizer=standard_normalizer,
        split_name="train"
    )
    
    # Check that sigma_data was updated
    assert config["model_params"]["sigma_data"] != initial_sigma
    assert isinstance(config["model_params"]["sigma_data"], float)
    assert config["model_params"]["sigma_data"] > 0


def test_get_diffusion_dataloaders(diffusion_config):
    """Test the get_diffusion_dataloaders function."""
    # Get dataloaders using the config with normalization
    train_loader, val_loader, test_loader = get_diffusion_dataloaders(
        diffusion_config, 
        num_workers=0  # Use 0 for testing
    )
    
    # Check types
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    
    # Check batch sizes
    assert train_loader.batch_size == diffusion_config["train_loader_params"]["batch_size"]
    
    # Check that loaders have data
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    
    # Check a batch
    batch = next(iter(train_loader))
    x, y = batch
    
    # Check types and shapes
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == y.shape
    assert x.shape[0] == train_loader.batch_size
    assert x.shape[1] == diffusion_config["model_params"]["channels"]


def test_diffusion_full_pipeline(diffusion_config):
    """Test the full pipeline from data loading to model input."""
    from src.diffusion.models.diffusion_factory import create_diffusion_model
    
    # Get dataloaders - use the config that already includes normalization
    train_loader, _, _ = get_diffusion_dataloaders(
        diffusion_config, 
        num_workers=0
    )
    
    # Create model
    model = create_diffusion_model(diffusion_config)
    
    # Get a batch
    x, y = next(iter(train_loader))
    
    # Forward pass (simulate training)
    loss = model(y)  # Use targets as clean images for training
    
    # Check loss
    assert torch.is_tensor(loss)
    assert loss.dim() == 0  # Scalar loss
    assert torch.isfinite(loss)  # Not NaN or inf
