import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MHDDataset(Dataset):
    def __init__(self, data_path, split="train", train_ratio=0.95, seed=42):
        """
        Dataset for MHD simulation data

        Args:
            data_path: Path to the numpy file with shape (num_samples, timesteps, nx, ny, channels)
            split: 'train', 'val', or 'test'
            train_ratio: Ratio of data to use for training
            seed: Random seed for reproducibility
        """
        self.data = np.load(data_path)
        self.num_samples, self.timesteps, self.nx, self.ny, self.channels = (
            self.data.shape
        )

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Create train/val/test indices
        indices = np.random.permutation(self.num_samples)
        train_size = int(train_ratio * self.num_samples)
        val_size = int((1 - train_ratio) / 2 * self.num_samples)

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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns:
            x: Initial condition with shape (channels=3, t=1, x=128, y=128)
            y: Full time evolution with shape (channels=3, t=101, x=128, y=128)
        """
        sample_idx = self.indices[idx]

        # Get full time evolution data
        data = self.data[
            sample_idx
        ]  # Shape: (timesteps=101, nx=128, ny=128, channels=3)

        # Initial condition is the first timestep
        ic = data[0:1]  # Shape: (1, nx=128, ny=128, channels=3)

        # TFNO3d expects: (batch, channels, t, x, y)
        # Need to permute from (t, x, y, channels) to (channels, t, x, y)
        x = torch.from_numpy(ic).float().permute(3, 0, 1, 2)  # Shape: (3, 1, 128, 128)
        y = (
            torch.from_numpy(data).float().permute(3, 0, 1, 2)
        )  # Shape: (3, 101, A 128, 128)

        return x, y


def get_dataloaders(data_path, batch_size=1, num_workers=4, train_ratio=0.95, seed=42):
    """Create dataloaders for train, validation, and test sets"""
    train_dataset = MHDDataset(
        data_path, split="train", train_ratio=train_ratio, seed=seed
    )
    val_dataset = MHDDataset(data_path, split="val", train_ratio=train_ratio, seed=seed)
    test_dataset = MHDDataset(
        data_path, split="test", train_ratio=train_ratio, seed=seed
    )

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
