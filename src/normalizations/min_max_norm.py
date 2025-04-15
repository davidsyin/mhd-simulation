"""Min-max normalization that scales data to a specified range."""

import torch
import torch.nn as nn

from .normalization_factory import register_normalization


class MinMaxNormalization(nn.Module):
    """
    Min-max normalization that scales data to a specified range.

    This normalization scales each channel independently to a specified range
    based on the provided per-channel min/max values.

    Args:
        min_val: Pre-computed minimum values per channel
        max_val: Pre-computed maximum values per channel
        feature_range: Tuple of (min, max) for the output range
        eps: Small value to avoid division by zero
    """

    def __init__(self, min_val=None, max_val=None, feature_range=(0, 1), eps=1e-8):
        super().__init__()
        self.feature_range = feature_range
        self.eps = eps

        # Register min and max values as buffers
        if min_val is not None:
            self.register_buffer("min_val", torch.tensor(min_val))
        else:
            self.min_val = None

        if max_val is not None:
            self.register_buffer("max_val", torch.tensor(max_val))
        else:
            self.max_val = None

        # # Debug prints
        # print(f"minval:{self.min_val}")
        # print(f"minval:{self.max_val}")
        # print(f"range:{self.feature_range}")

    def normalize(self, x):
        """Apply per-channel normalization to input"""
        if self.min_val is None or self.max_val is None:
            return x

        min_val = self.min_val.view(1, -1, 1, 1, 1)
        max_val = self.max_val.view(1, -1, 1, 1, 1)

        x_std = (x - min_val) / (max_val - min_val + self.eps)

        return (
            x_std * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )

    def denormalize(self, x):
        """Apply per-channel denormalization to output"""
        if self.min_val is None or self.max_val is None:
            return x

        min_val = self.min_val.view(1, -1, 1, 1, 1)
        max_val = self.max_val.view(1, -1, 1, 1, 1)

        x_std = (x - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )

        return x_std * (max_val - min_val + self.eps) + min_val


@register_normalization("minmax")
def create_minmax_normalization(norm_params):
    """Create a min-max normalization."""
    return MinMaxNormalization(
        min_val=norm_params.get("min_val"),
        max_val=norm_params.get("max_val"),
        feature_range=norm_params.get("feature_range", (0, 1)),
        eps=norm_params.get("eps", 1e-8),
    )
