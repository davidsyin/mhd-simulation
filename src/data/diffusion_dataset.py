import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ..normalizations import create_normalization
import math


class DiffusionDataset(Dataset):
    """Dataset for training diffusion models"""

    def __init__(
        self,
        data_path,
        config=None,
        input_normalizer=None,
        target_normalizer=None,
        split_name="train",
    ):
        """
        Args:
            data_path: Path to the data file
            config: Configuration for preprocessing
            input_normalizer: Normalizer for input data
            target_normalizer: Normalizer for target data
            split_name: Name of the split for logging ("train", "val", "test")
        """
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.split_name = split_name
        self.input_normalizer = input_normalizer
        self.target_normalizer = target_normalizer

        self.original_shape = None
        self.timesteps = None

        print(input_normalizer, target_normalizer)
        print(f"Loading {split_name} data from {data_path}")
        data = np.load(data_path, allow_pickle=True)
        data_dict = data.item() if isinstance(data, np.ndarray) else data

        self.inputs = data_dict["diff_inputs"]
        self.targets = data_dict["diff_targets"]

        self.original_shape = self.inputs.shape
        if len(self.original_shape) >= 3:
            # For shape (nsamples, channels, timesteps, nx, ny)
            self.timesteps = self.original_shape[2]

        print(f"Original {split_name} input shape: {self.inputs.shape}")
        print(f"Original {split_name} target shape: {self.targets.shape}")
        if self.timesteps:
            print(f"Timesteps per simulation: {self.timesteps}")

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

        normalize_after_prepare = self._should_normalize_after_prepare()

        print("prenormalized", self.inputs.shape, self.targets.shape)
        if not normalize_after_prepare:
            self._apply_normalization("all channels")

        # Then prepare data (reshape and select channels)
        if config:
            print("preparing", self.inputs.shape, self.targets.shape)
            self.inputs, self.targets = self.prepare_diffusion_input(
                self.inputs, self.targets, config
            )

            if normalize_after_prepare:
                self._apply_normalization("prepared channels")

            # Calculate sigma_data from normalized targets if this is the training set
            if split_name == "train" and "model_params" in config:
                self.calculate_sigma_data(config, split_name)

    def _normalizer_channels(self, normalizer):
        """Return the number of channels a normalizer is configured for."""
        if normalizer is None:
            return None

        for attr in ("mean", "std", "min_val", "max_val", "input_norm", "output_norm"):
            value = getattr(normalizer, attr, None)
            if value is not None:
                return int(value.numel())

        return None

    def _should_normalize_after_prepare(self):
        """Defer normalization when config values match selected channels, not raw channels."""
        raw_channels = self.inputs.shape[1]
        normalizer_channels = [
            self._normalizer_channels(self.input_normalizer),
            self._normalizer_channels(self.target_normalizer),
        ]

        return any(
            channel_count is not None and channel_count != raw_channels
            for channel_count in normalizer_channels
        )

    def _apply_normalization(self, scope):
        if self.input_normalizer:
            self.inputs = self.input_normalizer.normalize(self.inputs)
            print(f"Applied input normalization to {scope}")

        if self.target_normalizer:
            print("target, prenormalize", self.targets.shape)
            self.targets = self.target_normalizer.normalize(self.targets)
            print("targets, after", self.targets.shape)
            print(f"Applied target normalization to {scope}\n")

    def calculate_sigma_data(self, config, split_name):
        """Calculate sigma_data from normalized training targets for the selected channels"""
        if not hasattr(self, "targets") or self.targets is None:
            return

        targets = self.targets
        model_channels = config.get("model_params", {}).get("channels")

        if targets.ndim >= 3 and model_channels is not None and targets.shape[-1] == model_channels:
            channel_stds = [
                targets[..., c].std().item() for c in range(targets.shape[-1])
            ]
        else:
            channel_stds = [
                targets[:, c].std().item() for c in range(targets.shape[1])
            ]

        # Use mean of all channel stds as sigma_data
        sigma_data = float(torch.tensor(channel_stds).mean().item())
        print(
            f"Setting sigma_data={sigma_data} based on mean of std from normalized {split_name} set targets\n"
        )

        config["model_params"]["sigma_data"] = sigma_data

        return sigma_data

    def prepare_diffusion_input(self, x, y, config):
        """Prepare input data for diffusion model by reshaping from
        (nsamples, c=3, t, x, y) to (nsamples*t, c={1,2,3}, x, y)

        Note: This function expects already normalized data
        """
        model_params = config.get("model_params", {})
        dataset_params = config.get("dataset_params", {})

        out_channels = model_params.get("channels", 3)

        channel_indices = dataset_params.get("channel_indices", None)

        # Assert proper relationship between out_channels and channel_indices
        if channel_indices is not None:
            if isinstance(channel_indices, int):
                assert out_channels == 1, (
                    f"Model expects {out_channels} channels but only one channel index {channel_indices} was provided"
                )
            else:
                assert len(channel_indices) == out_channels, (
                    f"Model expects {out_channels} channels but {len(channel_indices)} channel indices were provided"
                )

        print("dataset", x.shape, y.shape)
        if len(x.shape) == 4:
            nsamples, in_channels, timesteps, n_points = x.shape
            print(in_channels)
            # _, out_channels, _, _, _ = y.shape

            # Reshape to (nsamples*t, c, n_points)
            x_reshaped = x.permute(0, 2, 1, 3).reshape(-1, in_channels, n_points)
            y_reshaped = y.permute(0, 2, 1, 3).reshape(-1, out_channels, n_points)

            # Handle channel selection
            print("indices", channel_indices)
            if channel_indices is not None:
                if isinstance(channel_indices, int):
                    channel_indices = [channel_indices]

                x_processed = x_reshaped[:, channel_indices, :]
                y_processed = y_reshaped[:, channel_indices, :]
            else:
                # if out_channels < in_channels:
                #     x_processed = x_reshaped[:, :out_channels, :]
                #     y_processed = y_reshaped[:, :out_channels, :]
                # else:
                #     x_processed = x_reshaped
                #     y_processed = y_reshaped
                x_processed = x_reshaped
                y_processed = y_reshaped
            x_processed = x_processed.permute(0,2,1)
            y_processed = y_processed.permute(0,2,1)

        else:        
            # Get spatial dimensions from data
            nx = x.shape[3] if len(x.shape) > 3 else 128
            ny = x.shape[4] if len(x.shape) > 4 else 128
    
            print(f"Preparing data with output channels={out_channels}")
            if channel_indices is not None:
                print(f"Using channel indices {channel_indices}")
    
            if len(x.shape) == 5:  # (nsamples, c, t, nx, ny)
                nsamples, in_channels, timesteps, nx, ny = x.shape
                _, out_channels, _, _, _ = y.shape
    
                # Reshape to (nsamples*t, c, nx, ny)
                x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(-1, in_channels, nx, ny)
                y_reshaped = y.permute(0, 2, 1, 3, 4).reshape(-1, out_channels, nx, ny)
    
                # Handle channel selection
                if channel_indices is not None:
                    if isinstance(channel_indices, int):
                        channel_indices = [channel_indices]
    
                    x_processed = x_reshaped[:, channel_indices, :, :]
                    y_processed = y_reshaped[:, channel_indices, :, :]
                else:
                    if out_channels < in_channels:
                        x_processed = x_reshaped[:, :out_channels, :, :]
                        y_processed = y_reshaped[:, :out_channels, :, :]
                    else:
                        x_processed = x_reshaped
                        y_processed = y_reshaped
            else:
                # If already in correct format, just pass through
                x_processed = x
                y_processed = y

        print("processed", x_processed.shape, y_processed.shape)
        print(f"Prepared {self.split_name} input shape: {x_processed.shape}")
        print(f"Prepared {self.split_name} target shape: {y_processed.shape}\n")

        return x_processed, y_processed

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Get input and target
        x = self.inputs[idx]
        y = self.targets[idx]

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        return x, y


def get_diffusion_dataloaders(config, num_workers=4):
    """Create dataloaders for diffusion model training

    Args:
        config: Configuration dictionary
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, test_loader
    """
    input_normalizer = None
    target_normalizer = None

    if "normalization_params" in config:
        norm_params = config.get("normalization_params", {})
        norm_type = norm_params.get("type", "identity").lower()

        inputs_stats_file = norm_params.get("inputs_stats_file")
        targets_stats_file = norm_params.get("targets_stats_file")

        # For backward compatibility, use the original stats_file if specific ones aren't provided
        if (
            not inputs_stats_file
            and not targets_stats_file
            and "stats_file" in norm_params
        ):
            inputs_stats_file = norm_params["stats_file"]
            targets_stats_file = norm_params["stats_file"]

        # Create input normalizer
        if inputs_stats_file and inputs_stats_file.endswith(".npz"):
            if not os.path.exists(inputs_stats_file):
                raise FileNotFoundError(
                    f"\nInput statistics file not found: {inputs_stats_file}"
                )

            print(f"\nLoading input normalization statistics from {inputs_stats_file}")
            input_stats = np.load(inputs_stats_file)

            input_norm_params = norm_params.copy()

            for key in input_stats.files:
                input_norm_params[key] = (
                    input_stats[key].tolist()
                    if hasattr(input_stats[key], "tolist")
                    else input_stats[key]
                )
                print(f"Input {key}: {input_stats[key]}")

            input_normalizer = create_normalization(
                {"normalization_params": input_norm_params}
            )
            print(f"Created {norm_type} normalizer for inputs\n")

        # Create target normalizer
        if targets_stats_file and targets_stats_file.endswith(".npz"):
            if not os.path.exists(targets_stats_file):
                raise FileNotFoundError(
                    f"\nTarget statistics file not found: {targets_stats_file}"
                )

            print(f"Loading target normalization statistics from {targets_stats_file}")
            target_stats = np.load(targets_stats_file)

            target_norm_params = norm_params.copy()

            for key in target_stats.files:
                target_norm_params[key] = (
                    target_stats[key].tolist()
                    if hasattr(target_stats[key], "tolist")
                    else target_stats[key]
                )
                print(f"Target {key}: {target_stats[key]}")

            target_normalizer = create_normalization(
                {"normalization_params": target_norm_params}
            )
            print(f"Created {norm_type} normalizer for targets\n")

        if not input_normalizer and not target_normalizer:
            direct_norm_params = norm_params.copy()

            if norm_type == "physics" and "input_norm" in direct_norm_params:
                direct_norm_params.setdefault(
                    "output_norm", direct_norm_params["input_norm"]
                )

            input_normalizer = create_normalization(
                {"normalization_params": direct_norm_params}
            )
            target_normalizer = create_normalization(
                {"normalization_params": direct_norm_params}
            )
            print(f"\nCreated {norm_type} normalizers from config values\n")

    dataset_params = config.get("dataset_params", {})
    train_path = dataset_params.get("train_path")
    val_path = dataset_params.get("val_path")
    test_path = dataset_params.get("test_path")

    if not train_path:
        raise ValueError("Train data path must be specified in config")

    train_batch_size = config.get("train_loader_params", {}).get("batch_size", 32)

    # Create datasets - create train first to calculate sigma_data
    print(f"Creating training dataset from {train_path}")
    train_dataset = DiffusionDataset(
        train_path,
        config=config,
        input_normalizer=input_normalizer,
        target_normalizer=target_normalizer,
        split_name="train",
    )

    val_dataset = None
    if val_path:
        print(f"Creating validation dataset from {val_path}")
        val_dataset = DiffusionDataset(
            val_path,
            config=config,
            input_normalizer=input_normalizer,
            target_normalizer=target_normalizer,
            split_name="validation",
        )

    test_dataset = None
    if test_path:
        print(f"Creating test dataset from {test_path}")
        test_dataset = DiffusionDataset(
            test_path,
            config=config,
            input_normalizer=input_normalizer,
            target_normalizer=target_normalizer,
            split_name="test",
        )

    # Create dataloaders with appropriate batch sizes
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=train_batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=True,
    #     persistent_workers=(num_workers > 0),
    # )
    train_loader = DataLoader(
        train_dataset,
        batch_size=26,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = None
    if val_dataset:
        # For validation, set batch size to timesteps
        val_batch_size = train_batch_size
        if val_dataset.timesteps:
            model_params = config.get("model_params", {})
            out_channels = model_params.get("channels", 3)
            val_batch_size = val_dataset.timesteps
            print(
                f"Using batch size {val_batch_size} for validation (= {val_dataset.timesteps} timesteps × {out_channels} channels)"
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,  # Full simulation in each batch if timesteps available
            shuffle=False,  # Important: keep temporal order
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    test_loader = None
    if test_dataset:
        # For test, set batch size to timesteps
        test_batch_size = train_batch_size
        if test_dataset.timesteps:
            model_params = config.get("model_params", {})
            out_channels = model_params.get("channels", 3)
            test_batch_size = test_dataset.timesteps
            print(
                f"Using batch size {test_batch_size} for test (= {test_dataset.timesteps} timesteps × {out_channels} channels)"
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,  # Full simulation in each batch if timesteps available
            shuffle=False,  # Important: keep temporal order
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    return train_loader, val_loader, test_loader
