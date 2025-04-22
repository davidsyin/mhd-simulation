"""Utility functions for visualization."""

from typing import Optional, Union


def safe_float_conversion(
    value: Union[str, float, int, None], default: Optional[float] = None
) -> Optional[float]:
    """Safely convert a value to float.

    Args:
        value: Value to convert
        default: Default value to use if conversion fails

    Returns:
        Converted float value or default
    """
    if value is None:
        return default

    try:
        return float(value)
    except (ValueError, TypeError):
        return default
