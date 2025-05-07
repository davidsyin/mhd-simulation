import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ...normalizations import create_normalization


class DiffusionDataset(Dataset):
    """Dataset for training diffusion models"""

    def __init__(self, data_path, config=None, transform=None, split_name="train"):
        """
        Args:
            data_path: Path to the data file
            config: Configuration for preprocessing
            transform: Additional transforms to apply
            split_name: Name of the split for logging ("train", "val", "test")
        """
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.transform = transform
        self.split_name = split_name

        print(f"Loading {split_name} data from {data_path}")
        # Load data
        data = np.load(data_path, allow_pickle=True)
        data_dict = data.item() if isinstance(data, np.ndarray) else data

        # Get inputs and targets
        self.inputs = data_dict["diff_inputs"]
        self.targets = data_dict["diff_targets"]

        print(f"Original {split_name} input shape: {self.inputs.shape}")
        print(f"Original {split_name} target shape: {self.targets.shape}")

        # Apply preprocessing
        if config:
            self.inputs, self.targets = self.preprocess(
                self.inputs, self.targets, config
            )

        # Create normalization
        if config and "normalization_params" in config:
            self.normalizer = create_normalization(config)
            print(
                f"Using {config['normalization_params']['type']} normalization for {split_name} data"
            )
        else:
            self.normalizer = None

    def preprocess(self, x, y, config):
        """Apply preprocessing for diffusion model data"""
        # Get parameters from config
        dataset_params = config.get("dataset_params", {})
        lb = dataset_params.get("lb", 1)
        lf = dataset_params.get("lf", 1)

        # Get spatial dimensions from data
        nx = x.shape[3] if len(x.shape) > 3 else dataset_params.get("nx", 128)
        ny = x.shape[4] if len(x.shape) > 4 else dataset_params.get("ny", 128)

        print(f"Processing with lb={lb}, lf={lf}, nx={nx}, ny={ny}")

        # Apply sliding window view for temporal data if needed
        if (
            dataset_params.get("use_sliding_window", False)
            and len(x.shape) > 2
            and x.shape[2] > lb
        ):
            print("Applying sliding window preprocessing...")
            from numpy.lib.stride_tricks import sliding_window_view

            # Apply sliding window to the time dimension (axis=2)
            # For data of shape (nsamples, channels, timesteps, nx, ny)
            x_processed = (
                sliding_window_view(x[:, :, lb - 1 :, :, :], window_shape=lf, axis=2)
                .transpose(0, 1, 2, 5, 3, 4)
                .reshape(-1, x.shape[1], lf, nx, ny)
            )

            y_processed = (
                sliding_window_view(y[:, :, lb - 1 :, :, :], window_shape=lf, axis=2)
                .transpose(0, 1, 2, 5, 3, 4)
                .reshape(-1, y.shape[1], lf, nx, ny)
            )
        else:
            # If no sliding window, just use the data as is
            x_processed = x
            y_processed = y

        print(f"Preprocessed {self.split_name} input shape: {x_processed.shape}")
        print(f"Preprocessed {self.split_name} target shape: {y_processed.shape}")

        return x_processed, y_processed

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Get input and target
        x = self.inputs[idx]
        y = self.targets[idx]

        # Convert to tensor if not already
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # Apply normalization if available
        if self.normalizer:
            # Make sure normalizer is on the same device as the data
            if hasattr(self.normalizer, "to"):
                data_device = x.device
                self.normalizer = self.normalizer.to(data_device)

            x = self.normalizer.normalize(x)
            y = self.normalizer.normalize(y)

        # Apply additional transforms
        if self.transform:
            x, y = self.transform(x, y)

        return x, y


def get_diffusion_dataloaders(config, num_workers=4):
    """Create dataloaders for diffusion model training

    Args:
        config: Configuration dictionary
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader
    """
    # Process normalization config and load NPZ stats if needed
    if "normalization_params" in config:
        norm_params = config["normalization_params"]

        if "stats_file" in norm_params and norm_params["stats_file"].endswith(".npz"):
            stats_file = norm_params["stats_file"]
            if not os.path.exists(stats_file):
                raise FileNotFoundError(f"Statistics file not found: {stats_file}")

            print(f"Loading normalization statistics from {stats_file}")
            stats = np.load(stats_file)

            # Create a copy of the config to avoid modifying the original
            norm_params_copy = norm_params.copy()

            for key in stats.files:
                norm_params_copy[key] = (
                    stats[key].tolist() if hasattr(stats[key], "tolist") else stats[key]
                )
                print(f"{key}: {stats[key]}")

            # Update the config with the loaded stats
            config["normalization_params"] = norm_params_copy

            # Set sigma_data from std in the stats file
            if "std" in stats.files and "model_params" in config:
                std_values = stats["std"]

                if isinstance(std_values, (list, np.ndarray)) and len(std_values) > 0:
                    # Use mean of all channel stds
                    sigma_data = float(np.mean(std_values))
                    print(
                        f"Setting sigma_data={sigma_data} based on mean of standard deviations from stats file"
                    )
                    config["model_params"]["sigma_data"] = sigma_data
        else:
            print(
                "\n***WARNING: Using normalization parameters directly from config file!***\n"
            )

    # Get data paths from config
    dataset_params = config.get("dataset_params", {})
    train_path = dataset_params.get("train_path")
    val_path = dataset_params.get("val_path")
    test_path = dataset_params.get("test_path")

    if not train_path:
        raise ValueError("Train data path must be specified in config")

    # Get batch sizes
    batch_size = config.get("train_loader_params", {}).get("batch_size", 32)

    # Create datasets
    print(f"Creating training dataset from {train_path}")
    train_dataset = DiffusionDataset(train_path, config=config, split_name="train")

    val_dataset = None
    if val_path:
        print(f"Creating validation dataset from {val_path}")
        val_dataset = DiffusionDataset(val_path, config=config, split_name="validation")

    test_dataset = None
    if test_path:
        print(f"Creating test dataset from {test_path}")
        test_dataset = DiffusionDataset(test_path, config=config, split_name="test")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    return train_loader, val_loader, test_loader
