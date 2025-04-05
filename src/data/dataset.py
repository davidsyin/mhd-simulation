import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from ..normalizations import create_normalization


class MHDDataset(Dataset):
    def __init__(
        self,
        data,
        normalization_config=None,
        split="train",
        train_ratio=0.95,
        seed=42,
    ):
        """
        Dataset for MHD simulation data with normalization support

        Args:
            data: Loaded numpy data with shape (num_samples, timesteps, nx, ny, channels)
            normalization_config: Configuration dict for normalization
            split: 'train', 'val', or 'test'
            train_ratio: Ratio of data to use for training
            seed: Random seed for reproducibility
        """
        # Extract shape information
        self.num_samples, self.timesteps, self.nx, self.ny, self.channels = data.shape

        # Reshape the whole dataset upfront
        # Convert from (num_samples, timesteps, nx, ny, channels) to
        # x_data: (num_samples, channels, 1, nx, ny) - initial conditions
        # y_data: (num_samples, channels, timesteps, nx, ny) - full time evolution
        self.x_data = torch.from_numpy(data[:, 0:1]).float().permute(0, 4, 1, 2, 3)
        self.y_data = torch.from_numpy(data).float().permute(0, 4, 1, 2, 3)

        # Create normalization module if specified
        self.normalizer = None
        if normalization_config:
            self.normalizer = create_normalization(normalization_config)
            # Debug print
            # norm_type = normalization_config.get("normalization_params", {}).get(
            #     "type", "unknown"
            # )
            # print(f"Using {norm_type} normalization at dataset level")

        # Set random seed for reproducibility
        np.random.seed(seed)
        # Debug print
        # print(f"Dataset using random seed: {seed} for {split} split")

        # Create train/val/test indices
        indices = np.random.permutation(self.num_samples)
        train_size = int(train_ratio * self.num_samples)
        val_size = math.ceil((1 - train_ratio) / 2 * self.num_samples)

        # Debug print
        #
        # print(f"indices:{indices}")
        # print(
        #     f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {self.num_samples - train_size - val_size}"
        # )

        if split == "train":
            self.indices = indices[:train_size]
        elif split == "val":
            self.indices = indices[train_size : train_size + val_size]
        elif split == "test":
            self.indices = indices[train_size + val_size :]
        else:
            raise ValueError(
                f"Split {split} not recognized. Use 'train', 'val', or 'test'"
            )

        print(f"Initialized {split} dataset with {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            x: Initial condition with shape (channels=3, t=1, x=128, y=128)
            y: Full time evolution with shape (channels=3, t=101, x=128, y=128)
        """
        sample_idx = self.indices[idx]
        x = self.x_data[sample_idx]
        y = self.y_data[sample_idx]

        # Apply normalization if specified
        if self.normalizer:
            # Add batch dimension for normalization
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            # # Debug print
            # print("Before norm:")
            # print("u", x[0, 0, 0, 0, 0])
            # print("v", x[0, 1, 0, 0, 0])
            # print("A", x[0, 2, 0, 0, 0])

            # Use the normalize method from the normalizer
            x = self.normalizer.normalize(x)
            y = self.normalizer.normalize(y)

            # # Debug print
            # print("After norm:")
            # print("u", x[0, 0, 0, 0, 0])
            # print("v", x[0, 1, 0, 0, 0])
            # print("A", x[0, 2, 0, 0, 0])

            # Remove batch dimension
            x = x.squeeze(0)
            y = y.squeeze(0)

        return x, y


def get_dataloaders(
    data_path,
    normalization_config=None,
    batch_size=1,
    num_workers=4,
    train_ratio=0.95,
    seed=42,
):
    """Create dataloaders for train, validation, and test sets with normalization support"""

    # Process normalization config and load NPZ stats if needed
    if normalization_config and "normalization_params" in normalization_config:
        norm_params = normalization_config["normalization_params"]

        # Check if there's a stats_file parameter
        if "stats_file" in norm_params and norm_params["stats_file"].endswith(".npz"):
            stats_file = norm_params["stats_file"]
            # Check if file exists
            if not os.path.exists(stats_file):
                raise FileNotFoundError(f"Statistics file not found: {stats_file}")

            print(f"Loading normalization statistics from {stats_file}")

            # Load the .npz file
            stats = np.load(stats_file)

            # Create a copy of the config to avoid modifying the original
            normalization_config = {"normalization_params": norm_params.copy()}

            # Update normalization parameters with loaded values
            for key in stats.files:
                normalization_config["normalization_params"][key] = (
                    stats[key].tolist() if hasattr(stats[key], "tolist") else stats[key]
                )
                print(f"{key}: {stats[key]}")

        else:
            print(
                "\n***WARNING: If they exist, statistics provided in the main config file will be used! Make sure you are providing correct stats for the correct normalization option!***\n"
            )

    data = np.load(data_path)
    print(f"\nCreating dataloaders with train_ratio={train_ratio}, seed={seed}")

    # Create datasets with the processed normalization config
    train_dataset = MHDDataset(
        data,
        normalization_config=normalization_config,
        split="train",
        train_ratio=train_ratio,
        seed=seed,
    )
    val_dataset = MHDDataset(
        data,
        normalization_config=normalization_config,
        split="val",
        train_ratio=train_ratio,
        seed=seed,
    )
    test_dataset = MHDDataset(
        data,
        normalization_config=normalization_config,
        split="test",
        train_ratio=train_ratio,
        seed=seed,
    )

    # Create and return dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
