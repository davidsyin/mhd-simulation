"""Min-max normalization that scales data to [0,1] range."""

import torch
import torch.nn as nn

from .normalization_factory import register_normalization


class MinMaxNormalization(nn.Module):
    """
    Min-max normalization that scales data to a specified range.

    This normalization computes global min and max values for each channel
    and scales the data to the specified range (default [0,1]).

    Args:
        feature_range: Tuple of (min, max) for scaled output range
        per_channel: Whether to normalize each channel separately
        eps: Small value to avoid division by zero
    """

    def __init__(self, feature_range=(0, 1), per_channel=True, eps=1e-8):
        super().__init__()
        self.feature_range = feature_range
        self.per_channel = per_channel
        self.eps = eps

        # These will be initialized during first call
        self.register_buffer("data_min", None)
        self.register_buffer("data_max", None)

    def fit(self, x):
        """Compute min and max values from data"""
        if self.per_channel:
            # Compute per-channel min/max
            dims = tuple(range(2, x.dim()))  # Spatial dimensions
            min_val = x.min(dim=0, keepdim=True)[0].min(dims, keepdim=True)[0]
            max_val = x.max(dim=0, keepdim=True)[0].max(dims, keepdim=True)[0]
        else:
            # Global min/max
            min_val = x.min().expand_as(x[[0]])
            max_val = x.max().expand_as(x[[0]])

        self.register_buffer("data_min", min_val)
        self.register_buffer("data_max", max_val)

    def normalize(self, x):
        """Apply normalization to input"""
        if self.data_min is None or self.data_max is None:
            self.fit(x)

        x_std = (x - self.data_min) / (self.data_max - self.data_min + self.eps)
        x_scaled = (
            x_std * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )

        return x_scaled

    def denormalize(self, x):
        """Apply denormalization to output"""
        if self.data_min is None or self.data_max is None:
            raise RuntimeError("Cannot denormalize before calling normalize() first")

        x_std = (x - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )
        x_orig = x_std * (self.data_max - self.data_min) + self.data_min

        return x_orig


@register_normalization("minmax")
def create_minmax_normalization(norm_params):
    """Create a min-max normalization."""
    return MinMaxNormalization(
        feature_range=norm_params.get("feature_range", (0, 1)),
        per_channel=norm_params.get("per_channel", True),
        eps=norm_params.get("eps", 1e-8),
    )
