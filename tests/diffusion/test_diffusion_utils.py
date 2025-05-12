"""Tests for diffusion utility functions."""

import pytest
import torch
import numpy as np

from src.diffusion.utils.utils import (
    exists,
    default,
    once,
    cast_tuple,
    divisible_by,
    log,
)


def test_exists():
    """Test the exists utility function."""
    assert exists(1) is True
    assert exists(0) is True
    assert exists([]) is True
    assert exists("") is True
    assert exists(None) is False


def test_default():
    """Test the default utility function."""
    assert default(None, 1) == 1
    assert default(0, 1) == 0
    assert default("", "default") == ""
    assert default(None, lambda: "computed") == "computed"

    # Test with callable default
    counter = [0]

    def increment():
        counter[0] += 1
        return counter[0]

    # Should call the function only when val is None
    assert default(None, increment) == 1
    assert default(None, increment) == 2
    assert default("value", increment) == "value"
    assert counter[0] == 2  # Function should have been called twice


def test_once():
    """Test the once decorator utility function."""
    counter = [0]

    @once
    def increment_once(x):
        counter[0] += 1
        return counter[0]

    # First call should work
    assert increment_once(1) == 1
    assert counter[0] == 1

    # Subsequent calls should do nothing
    assert increment_once(2) is None
    assert increment_once(3) is None
    assert counter[0] == 1  # Counter shouldn't have increased


def test_cast_tuple():
    """Test the cast_tuple utility function."""
    assert cast_tuple(1, 3) == (1, 1, 1)
    assert cast_tuple((1, 2), 3) == (1, 2)
    assert cast_tuple((1, 2, 3), 3) == (1, 2, 3)
    assert cast_tuple([1, 2], 3) == ([1, 2],) * 3  # Lists remain as is, but repeated


def test_divisible_by():
    """Test the divisible_by utility function."""
    assert divisible_by(10, 2) is True
    assert divisible_by(10, 3) is False
    assert divisible_by(0, 5) is True


def test_log():
    """Test the log utility function."""
    x = torch.tensor([1.0, 2.0, 3.0])
    expected = torch.log(x)
    assert torch.allclose(log(x), expected)

    # Test with small values
    x_small = torch.tensor([1e-30, 1e-100])
    # Should not produce -inf
    result = log(x_small)
    assert not torch.isinf(result).any()
