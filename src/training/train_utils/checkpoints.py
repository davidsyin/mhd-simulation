"""Functions for saving and loading model checkpoints during training."""

import os
from collections.abc import Mapping, Sequence

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    scheduler,
    ckpt_path,
    denorm_loss_rel_l2=None,
    denorm_loss_mse=None,
):
    """Save checkpoint

    Args:
        model: Neural network model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss value
        scheduler: Learning rate scheduler
        ckpt_path: Path to save checkpoint
        denorm_loss_rel_l2: Denormalized loss value-- Relative L2 (optional)
        denorm_loss_mse: Denormalized loss value-- MSE (optional)
    """
    # If using DDP, save only the model's module
    if isinstance(model, DDP):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    print("Saving at", ckpt_path)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }

    # Add denormalized losses if provided
    if denorm_loss_rel_l2 is not None:
        checkpoint_data["denorm_loss_rel_l2"] = denorm_loss_rel_l2
    if denorm_loss_mse is not None:
        checkpoint_data["denorm_loss_mse"] = denorm_loss_mse

    torch.save(checkpoint_data, ckpt_path)


def _normalize_input_channel_map(input_channel_map):
    """Normalize supported channel-map formats to (checkpoint_idx, model_idx)."""
    if input_channel_map is None:
        return []

    if isinstance(input_channel_map, Mapping):
        return [(int(src), int(dst)) for src, dst in input_channel_map.items()]

    if not isinstance(input_channel_map, Sequence) or isinstance(input_channel_map, str):
        raise TypeError(
            "input_channel_map must be a mapping, a list of pairs, or a list of "
            "checkpoint channel indices ordered by model channel."
        )

    channel_map = list(input_channel_map)
    if not channel_map:
        return []

    if all(isinstance(item, int) for item in channel_map):
        return [(int(src), model_idx) for model_idx, src in enumerate(channel_map)]

    pairs = []
    for item in channel_map:
        if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
            raise TypeError(
                "input_channel_map entries must be [checkpoint_channel, model_channel] pairs."
            )
        pairs.append((int(item[0]), int(item[1])))
    return pairs


def _can_map_input_channels(checkpoint_tensor, model_tensor):
    if checkpoint_tensor.ndim < 2 or checkpoint_tensor.ndim != model_tensor.ndim:
        return False

    return all(
        checkpoint_tensor.shape[dim] == model_tensor.shape[dim]
        for dim in range(checkpoint_tensor.ndim)
        if dim != 1
    )


def _copy_input_channels(name, checkpoint_tensor, model_tensor, input_channel_pairs):
    adapted_tensor = model_tensor.detach().clone()
    checkpoint_tensor = checkpoint_tensor.to(
        device=adapted_tensor.device, dtype=adapted_tensor.dtype
    )

    for checkpoint_channel, model_channel in input_channel_pairs:
        if checkpoint_channel < 0 or checkpoint_channel >= checkpoint_tensor.shape[1]:
            raise ValueError(
                f"{name}: checkpoint input channel {checkpoint_channel} is out of "
                f"range for tensor shape {tuple(checkpoint_tensor.shape)}."
            )
        if model_channel < 0 or model_channel >= adapted_tensor.shape[1]:
            raise ValueError(
                f"{name}: model input channel {model_channel} is out of range for "
                f"tensor shape {tuple(adapted_tensor.shape)}."
            )

        adapted_tensor.select(1, model_channel).copy_(
            checkpoint_tensor.select(1, checkpoint_channel)
        )

    return adapted_tensor


def _matches_input_channel_keys(key, input_channel_keys):
    if input_channel_keys is None:
        return True

    if isinstance(input_channel_keys, str):
        input_channel_keys = [input_channel_keys]

    return any(pattern in key for pattern in input_channel_keys)


def _prepare_state_dict_for_load(
    checkpoint_state_dict,
    model_state_dict,
    input_channel_map=None,
    input_channel_keys=None,
    skip_mismatched=False,
):
    input_channel_pairs = _normalize_input_channel_map(input_channel_map)
    prepared_state_dict = {}
    remapped_keys = []
    skipped_keys = []

    for key, checkpoint_tensor in checkpoint_state_dict.items():
        model_tensor = model_state_dict.get(key)
        if model_tensor is None:
            prepared_state_dict[key] = checkpoint_tensor
            continue

        if checkpoint_tensor.shape == model_tensor.shape:
            prepared_state_dict[key] = checkpoint_tensor
            continue

        if (
            input_channel_pairs
            and _matches_input_channel_keys(key, input_channel_keys)
            and _can_map_input_channels(checkpoint_tensor, model_tensor)
        ):
            prepared_state_dict[key] = _copy_input_channels(
                key, checkpoint_tensor, model_tensor, input_channel_pairs
            )
            remapped_keys.append(key)
            continue

        if skip_mismatched:
            skipped_keys.append(
                (key, tuple(checkpoint_tensor.shape), tuple(model_tensor.shape))
            )
            continue

        prepared_state_dict[key] = checkpoint_tensor

    return prepared_state_dict, remapped_keys, skipped_keys


def load_checkpoint(
    ckpt_path,
    model,
    optimizer=None,
    scheduler=None,
    input_channel_map=None,
    input_channel_keys=None,
    strict=True,
    skip_mismatched=None,
):
    """Load checkpoint

    Args:
        ckpt_path: Path to checkpoint
        model: Neural network model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        input_channel_map: Optional mapping from checkpoint input channels to
            model input channels. Supported formats:
            - {checkpoint_channel: model_channel}
            - [[checkpoint_channel, model_channel], ...]
            - [checkpoint_channel, ...], interpreted as checkpoint channels
              ordered by model channel.
        input_channel_keys: Optional string or list of strings. When provided,
            input channel mapping is only applied to checkpoint keys containing
            one of these substrings.
        strict: Passed to load_state_dict. Keep True for exact resume; use False
            for partial fine-tuning loads.
        skip_mismatched: If True, shape-mismatched tensors that cannot be
            channel-mapped are left at the model initialization. Defaults to
            the inverse of strict.

    Returns:
        dict: Checkpoint data with epoch, loss, and denorm_loss if available
    """
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} does not exist. Starting from scratch.")
        return {
            "epoch": 0,
            "loss": float("inf"),
            "denorm_loss_rel_l2": float("inf"),
            "denorm_loss_mse": float("inf"),
        }

    model_to_load = model.module if isinstance(model, DDP) else model
    map_location = next(model_to_load.parameters()).device
    checkpoint = torch.load(
        ckpt_path, map_location=map_location, weights_only=False
    )

    # Clean up the model state dict if _metadata is present
    if "_metadata" in checkpoint["model_state_dict"]:
        del checkpoint["model_state_dict"]["_metadata"]

    if skip_mismatched is None:
        skip_mismatched = not strict

    state_dict, remapped_keys, skipped_keys = _prepare_state_dict_for_load(
        checkpoint["model_state_dict"],
        model_to_load.state_dict(),
        input_channel_map=input_channel_map,
        input_channel_keys=input_channel_keys,
        skip_mismatched=skip_mismatched,
    )

    if remapped_keys:
        print(
            "Remapped checkpoint input channels for "
            f"{len(remapped_keys)} tensor(s): {remapped_keys}"
        )
    if skipped_keys:
        print("Skipped shape-mismatched checkpoint tensor(s):")
        for key, checkpoint_shape, model_shape in skipped_keys:
            print(f"- {key}: checkpoint {checkpoint_shape} -> model {model_shape}")

    # Handle DDP model
    model_to_load.load_state_dict(state_dict, strict=strict)

    # if optimizer:
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # if scheduler and checkpoint.get("scheduler_state_dict"):
    #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Return all checkpoint data as a dictionary
    result = {
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"],
        "denorm_loss_rel_l2": checkpoint.get("denorm_loss_rel_l2", float("inf")),
        "denorm_loss_mse": checkpoint.get("denorm_loss_mse", float("inf")),
    }

    return result
