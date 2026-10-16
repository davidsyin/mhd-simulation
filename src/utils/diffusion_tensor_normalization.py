"""Utility functions for tensor denormalization with channel mapping."""

import torch


def _normalizer_channels(normalizer):
    for attr in ("mean", "std", "min_val", "max_val", "input_norm", "output_norm"):
        value = getattr(normalizer, attr, None)
        if value is not None:
            return int(value.numel())

    return None


def denormalize_with_channel_mapping(tensor, normalizer, channel_indices):
    """
    Denormalizes tensors with proper channel mapping to handle different channel counts.

    This function handles the case where normalizer expects a specific number of channels
    (typically 3) but the input tensor may have fewer channels.

    Args:
        tensor: Input tensor to denormalize with shape [batch_size, channels, nx, ny]
        normalizer: Normalizer object with denormalize method
        channel_indices: Indices specifying which channels in normalizer space the tensor channels map to

    Returns:
        Denormalized tensor with same shape as input
    """
    if normalizer is None:
        return tensor

    if tensor.ndim < 2:
        raise ValueError(
            "tensor must have at least batch and channel dimensions, "
            f"got shape {tuple(tensor.shape)}"
        )

    normalizer_channels = _normalizer_channels(normalizer)
    if normalizer_channels is None:
        return normalizer.denormalize(tensor)

    if isinstance(channel_indices, int):
        channel_indices = [channel_indices]
    elif channel_indices is None:
        channel_indices = list(range(normalizer_channels))

    if tensor.shape[1] in (normalizer_channels, len(channel_indices)):
        channel_axis = 1
    elif tensor.shape[-1] in (normalizer_channels, len(channel_indices)):
        channel_axis = tensor.ndim - 1
    else:
        raise ValueError(
            "could not identify channel dimension for tensor shape "
            f"{tuple(tensor.shape)} with normalizer_channels={normalizer_channels} "
            f"and channel_indices={channel_indices}"
        )

    channel_first = (
        tensor if channel_axis == 1 else torch.movedim(tensor, channel_axis, 1)
    )
    bs, c, *spatial_dims = channel_first.shape

    if normalizer_channels == c:
        denormalized = normalizer.denormalize(channel_first)
        return (
            denormalized
            if channel_axis == 1
            else torch.movedim(denormalized, 1, channel_axis)
        )

    if max(channel_indices, default=-1) >= normalizer_channels:
        raise ValueError(
            f"channel_indices {channel_indices} are incompatible with a "
            f"{normalizer_channels}-channel normalizer"
        )

    # Add the singleton time dimension expected by dataset-level normalizers.
    tensor_5d = channel_first.unsqueeze(2)  # (bs, c, 1, ...)
    full_tensor = torch.zeros(
        bs,
        normalizer_channels,
        1,
        *spatial_dims,
        device=tensor.device,
        dtype=tensor.dtype,
    )

    for i, channel_idx in enumerate(channel_indices):
        if i < c:
            full_tensor[:, channel_idx] = tensor_5d[:, i]

    denormalized = normalizer.denormalize(full_tensor)

    result = torch.zeros(
        bs, c, *spatial_dims, device=tensor.device, dtype=tensor.dtype
    )
    for i, channel_idx in enumerate(channel_indices):
        if i < c:
            result[:, i] = denormalized[:, channel_idx, 0]

    return result if channel_axis == 1 else torch.movedim(result, 1, channel_axis)
