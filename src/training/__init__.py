"""Training utilities for neural operator and diffusion models."""

__all__ = ["train_neural_operator", "train_diffusion"]


def train_neural_operator(*args, **kwargs):
    from .train_neural_operator import train

    return train(*args, **kwargs)


def train_diffusion(*args, **kwargs):
    from .train_diffusion import train

    return train(*args, **kwargs)
