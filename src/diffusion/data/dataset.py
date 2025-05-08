import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ...normalizations import create_normalization


class DiffusionDataset(Dataset):
    """Dataset for training diffusion models"""

    def __init__(self, data_path, config=None, split_name="train"):
        """
        Args:
            data_path: Path to the data file
            config: Configuration for preprocessing
            split_name: Name of the split for logging ("train", "val", "test")
        """
        super().__init__()
        self.data_path = data_path
        self.config = config
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

        # Create normalization first (before preparing diffusion input)
        if config and "normalization_params" in config:
            self.normalizer = create_normalization(config)
            print(
                f"Using {config['normalization_params']['type']} normalization for {split_name} data"
            )
        else:
            self.normalizer = None

        # Apply data preparation
        if config:
            self.inputs, self.targets = self.prepare_diffusion_input(
                self.inputs, self.targets, config
            )

            # Calculate sigma_data from normalized training targets if this is the training set
            if split_name == "train" and "model_params" in config:
                self.calculate_sigma_data(config)

    def calculate_sigma_data(self, config):
        """Calculate sigma_data from normalized training targets for the selected channels"""
        if not hasattr(self, "targets") or self.targets is None:
            return

        # We can use targets directly as they're already normalized by prepare_diffusion_input
        targets = self.targets

        # Calculate standard deviation
        # For each channel, calculate std across all samples, spatial dimensions
        channel_stds = []

        for c in range(targets.shape[1]):  # Iterate over channels
            # Calculate std for this channel across all other dimensions
            channel_std = torch.tensor(targets[:, c]).std().item()
            channel_stds.append(channel_std)
            print(f"Channel {c} standard deviation: {channel_std:.6f}")

        # Use mean of all channel stds as sigma_data
        sigma_data = float(np.mean(channel_stds))
        print(
            f"Setting sigma_data={sigma_data} based on mean of standard deviations from normalized targets"
        )

        # Update config
        config["model_params"]["sigma_data"] = sigma_data

        return sigma_data

    def prepare_diffusion_input(self, x, y, config):
        """Prepare input data for diffusion model by reshaping from
        (nsamples, c=3, t, x, y) to (nsamples*t, c={1,2,3}, x, y)
        """
        # Get parameters from config
        model_params = config.get("model_params", {})
        dataset_params = config.get("dataset_params", {})

        # Get number of channels from model config
        out_channels = model_params.get("channels", 3)

        # Get channel indices to use (if specified)
        # Can be a single index (int) or a list of indices
        channel_indices = dataset_params.get("channel_indices", None)

        # Assert proper relationship between out_channels and channel_indices
        if channel_indices is not None:
            if isinstance(channel_indices, int):
                # If channel_indices is an int, make sure out_channels is 1
                assert out_channels == 1, (
                    f"Model expects {out_channels} channels but only one channel index {channel_indices} was provided"
                )
            else:
                # If channel_indices is a list, make sure its length matches out_channels
                assert len(channel_indices) == out_channels, (
                    f"Model expects {out_channels} channels but {len(channel_indices)} channel indices were provided"
                )

        # Get spatial dimensions from data
        nx = x.shape[3] if len(x.shape) > 3 else 128
        ny = x.shape[4] if len(x.shape) > 4 else 128

        print(f"Preparing data with output channels={out_channels}")
        if channel_indices is not None:
            print(f"Using channel indices {channel_indices}")

        if len(x.shape) == 5:  # (nsamples, c, t, nx, ny)
            nsamples, in_channels, timesteps, nx, ny = x.shape

            # Reshape to (nsamples*t, c, nx, ny)
            x_reshaped = x.transpose(0, 2, 1, 3, 4).reshape(-1, in_channels, nx, ny)
            y_reshaped = y.transpose(0, 2, 1, 3, 4).reshape(-1, in_channels, nx, ny)

            # Handle channel selection
            if channel_indices is not None:
                # Convert to list if it's a single integer
                if isinstance(channel_indices, int):
                    channel_indices = [channel_indices]

                # Extract the specified channels
                x_processed = x_reshaped[:, channel_indices, :, :]
                y_processed = y_reshaped[:, channel_indices, :, :]

                # Check if the selected channels match the expected output channels
                if len(channel_indices) != out_channels:
                    print(
                        f"WARNING: Number of selected channels ({len(channel_indices)}) "
                        f"doesn't match model channels ({out_channels})"
                    )
            else:
                # If no specific channels are selected
                if out_channels < in_channels:
                    # By default, take the first 'out_channels' channels
                    x_processed = x_reshaped[:, :out_channels, :, :]
                    y_processed = y_reshaped[:, :out_channels, :, :]
                else:
                    x_processed = x_reshaped
                    y_processed = y_reshaped
        else:
            # If already in correct format, just pass through
            x_processed = x
            y_processed = y

        # Apply normalization to the processed data (after channel selection)
        if self.normalizer:
            x_processed = self.normalizer.normalize(torch.tensor(x_processed)).numpy()
            y_processed = self.normalizer.normalize(torch.tensor(y_processed)).numpy()

        print(f"Prepared {self.split_name} input shape: {x_processed.shape}")
        print(f"Prepared {self.split_name} target shape: {y_processed.shape}")

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

        # No normalization here - it has already been applied during prepare_diffusion_input

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

            # Note: We no longer set sigma_data here from stats file
            # It will be calculated correctly in the DiffusionDataset after normalization

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

    # Create datasets - create train first to calculate sigma_data
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
