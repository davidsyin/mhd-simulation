"""Loss functions and factory for creating them from configuration."""

from .loss_factory import create_loss, register_loss, LOSS_REGISTRY
from .lp_loss import LpLoss
from .physics_informed import MHDVecPotLoss

__all__ = ["create_loss", "register_loss", "LOSS_REGISTRY", "LpLoss", "MHDVecPotLoss"]
