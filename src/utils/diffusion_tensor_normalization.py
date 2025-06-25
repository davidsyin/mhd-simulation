"""Utility functions for tensor denormalization with channel mapping."""

import torch


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

    # Get tensor shape
    bs, c, *spatial_dims = tensor.shape

    # First add the singleton time dimension required by the normalizer
    tensor_5d = tensor.unsqueeze(2)  # (bs, c, 1, nx, ny)

    # Create a full 3-channel tensor with zeros
    full_tensor = torch.zeros(bs, 3, 1, *spatial_dims, device=tensor.device)

    # Place each channel in its correct position according to channel_indices
    for i, channel_idx in enumerate(channel_indices):
        if i < c:  # Only use as many channels as we have in our tensor
            full_tensor[:, channel_idx] = tensor_5d[:, i]

    # Apply denormalization to the full tensor
    denormalized = normalizer.denormalize(full_tensor)

    # Extract only the channels we need based on original indices
    result = torch.zeros(bs, c, *spatial_dims, device=tensor.device)
    for i, channel_idx in enumerate(channel_indices):
        if i < c:  # Only extract as many channels as we need
            result[:, i] = denormalized[:, channel_idx, 0]  # Remove time dimension

    return result
