"""
Training utilities for neural operator and diffusion models.
"""

from .train_neural_operator import train as train_neural_operator
from .train_diffusion import train as train_diffusion

__all__ = ["train_neural_operator", "train_diffusion"]
