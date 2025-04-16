"""Base class for MHD loss functions."""

import numpy as np
import torch
import torch.nn.functional as F
import math
from .lp_loss import LpLoss


class LossMHD(object):
    """Calculate loss for MHD equations with magnetic field, using original derivatives in paper."""

    def __init__(
        self,
        nu=1e-4,
        eta=1e-4,
        rho0=1.0,
        data_weight=1.0,
        ic_weight=1.0,
        pde_weight=1.0,
        constraint_weight=1.0,
        use_data_loss=True,
        use_ic_loss=True,
        use_pde_loss=True,
        use_constraint_loss=True,
        u_weight=1.0,
        v_weight=1.0,
        Bx_weight=1.0,
        By_weight=1.0,
        Du_weight=1.0,
        Dv_weight=1.0,
        DBx_weight=1.0,
        DBy_weight=1.0,
        div_B_weight=1.0,
        div_vel_weight=1.0,
        Lx=1.0,
        Ly=1.0,
        tend=1.0,
        use_weighted_mean=False,
        **kwargs,
    ):
        self.nu = nu
        self.eta = eta
        self.rho0 = rho0
        self.data_weight = data_weight
        self.ic_weight = ic_weight
        self.pde_weight = pde_weight
        self.constraint_weight = constraint_weight
        self.use_data_loss = use_data_loss
        self.use_ic_loss = use_ic_loss
        self.use_pde_loss = use_pde_loss
        self.use_constraint_loss = use_constraint_loss
        self.u_weight = u_weight
        self.v_weight = v_weight
        self.Bx_weight = Bx_weight
        self.By_weight = By_weight
        self.Du_weight = Du_weight
        self.Dv_weight = Dv_weight
        self.DBx_weight = DBx_weight
        self.DBy_weight = DBy_weight
        self.div_B_weight = div_B_weight
        self.div_vel_weight = div_vel_weight
        self.Lx = Lx
        self.Ly = Ly
        self.tend = tend
        self.use_weighted_mean = use_weighted_mean

        if not self.use_data_loss:
            self.data_weight = 0
        if not self.use_ic_loss:
            self.ic_weight = 0
        if not self.use_pde_loss:
            self.pde_weight = 0
        if not self.use_constraint_loss:
            self.constraint_weight = 0

    def __call__(self, pred, true, inputs, return_loss_dict=False):
        if not return_loss_dict:
            loss = self.compute_loss(pred, true, inputs)
            return loss
        else:
            loss, loss_dict = self.compute_losses(pred, true, inputs)
            return loss, loss_dict

    def Du_t(self, u, dt):
        """Compute time derivative"""
        u_t = (u[:, 2:] - u[:, :-2]) / (2 * dt)
        return u_t

    def Lap(self, u_h, k_i, k_j):
        """Helper method to calculate laplacian of variable"""
        lap = -(k_i**2 + k_j**2)
        u_lap_h = lap * u_h
        return u_lap_h

    def Du_i(self, u_h, k_i):
        u_i_h = (1j * k_i) * u_h
        return u_i_h

    def Du_ij(self, u_h, k_i, k_j):
        u_ij_h = (1j * k_i) * (1j * k_j) * u_h
        return u_ij_h

    def Du_ii(self, u_h, k_i):
        u_ii_h = self.Du_ij(u_h, k_i, k_i)
        return u_ii_h
