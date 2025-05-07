from functools import wraps
import torch
import math


def exists(val):
    """Check if a value exists (is not None)."""
    return val is not None


def default(val, d):
    """Return val if it exists, otherwise return default d."""
    if exists(val):
        return val
    return d() if callable(d) else d


def once(fn):
    """Decorator to ensure a function is only called once."""
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def cast_tuple(t, length=1):
    """Cast input to a tuple of specified length."""
    if isinstance(t, tuple):
        return t
    return (t,) * length


def divisible_by(numer, denom):
    """Check if number is divisible by another."""
    return (numer % denom) == 0


def log(t, eps=1e-20):
    """Safe logarithm with minimum clamping to avoid numerical issues."""
    return torch.log(t.clamp(min=eps))


# normalization functions
def normalize_to_neg_one_to_one(img):
    """Normalize image from [0,1] to [-1,1]."""
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    """Unnormalize from [-1,1] to [0,1]."""
    return (t + 1) * 0.5
