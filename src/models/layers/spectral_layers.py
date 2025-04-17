"""Spectral layer implementations for Fourier-based neural operators."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import tltorch


def fourier_derivatives(x: Tensor, ell: List[float]) -> Tuple[Tensor, Tensor]:
    """
    Fourier derivative function for PINO

    Parameters
    ----------
    x : Tensor
        Input tensor
    ell : List[float]
        Domain lengths

    Returns
    -------
    Tuple[Tensor, Tensor]
        First and second derivatives
    """

    # check that input shape matches domain length
    if len(x.shape) - 2 != len(ell):
        raise ValueError("input shape doesn't match domain dims")

    # set pi from numpy
    pi = float(np.pi)

    # get needed dims
    n = x.shape[2:]
    dim = len(ell)

    # get device
    device = x.device

    # compute fourier transform
    x_h = torch.fft.fftn(x, dim=list(range(2, dim + 2)))

    # make wavenumbers
    k_x = []
    for i, nx in enumerate(n):
        k_x.append(
            torch.cat(
                (
                    torch.arange(start=0, end=nx // 2, step=1, device=device),
                    torch.arange(start=-nx // 2, end=0, step=1, device=device),
                ),
                0,
            ).reshape((i + 2) * [1] + [nx] + (dim - i - 1) * [1])
        )

    # compute laplacian in fourier space
    j = torch.complex(
        torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
    )  # Cuda graphs does not work here
    wx_h = [j * k_x_i * x_h * (2 * pi / ell[i]) for i, k_x_i in enumerate(k_x)]
    wxx_h = [
        j * k_x_i * wx_h_i * (2 * pi / ell[i])
        for i, (wx_h_i, k_x_i) in enumerate(zip(wx_h, k_x))
    ]

    # inverse fourier transform out
    wx = torch.cat(
        [torch.fft.ifftn(wx_h_i, dim=list(range(2, dim + 2))).real for wx_h_i in wx_h],
        dim=1,
    )
    wxx = torch.cat(
        [
            torch.fft.ifftn(wxx_h_i, dim=list(range(2, dim + 2))).real
            for wxx_h_i in wxx_h
        ],
        dim=1,
    )
    return (wx, wxx)


def fourier_derivatives_lap(x: Tensor, ell: List[float]) -> Tensor:
    """
    Fourier derivative laplacian function

    Parameters
    ----------
    x : Tensor
        Input tensor
    ell : List[float]
        Domain lengths

    Returns
    -------
    Tensor
        Laplacian of input
    """

    # check that input shape matches domain length
    if len(x.shape) - 2 != len(ell):
        raise ValueError("input shape doesn't match domain dims")

    # set pi from numpy
    pi = float(np.pi)

    # get needed dims
    n = x.shape[2:]
    dim = len(ell)

    # get device
    device = x.device

    # compute fourier transform
    x_h = torch.fft.fftn(x, dim=list(range(2, dim + 2)))

    # make wavenumbers
    k_x = []
    for i, nx in enumerate(n):
        k_x.append(
            (2 * pi / ell[i])
            * torch.cat(
                (
                    torch.arange(start=0, end=nx // 2, step=1, device=device),
                    torch.arange(start=-nx // 2, end=0, step=1, device=device),
                ),
                0,
            ).reshape((i + 2) * [1] + [nx] + (dim - i - 1) * [1])
        )
    lap = torch.zeros_like(k_x[0])
    for i in k_x:
        lap = lap - i**2

    # compute laplacian in fourier space
    wx_h = lap * x_h

    # inverse fourier transform out
    wx = torch.fft.ifftn(wx_h, dim=list(range(2, dim + 2))).real
    return wx


def fourier_derivatives_ptot(
    p: Tensor,
    div_vel_grad_vel: Tensor,
    div_B_grad_B: Tensor,
    B2_h: Tensor,
    rho0: float,
    ell: List[float],
) -> List[Tensor]:
    """
    Fourier derivative function to calculate ptot in MHD equations

    Parameters
    ----------
    p : Tensor
        Pressure tensor
    div_vel_grad_vel : Tensor
        Divergence of velocity gradient tensor
    div_B_grad_B : Tensor
        Divergence of magnetic field gradient tensor
    B2_h : Tensor
        Square of magnetic field in Fourier space
    rho0 : float
        Reference density
    ell : List[float]
        Domain lengths

    Returns
    -------
    List[Tensor]
        Derivatives of total pressure
    """

    # check that input shape matches domain length
    if len(div_vel_grad_vel.shape) - 2 != len(ell):
        raise ValueError("input shape doesn't match domain dims")

    # set pi from numpy
    pi = float(np.pi)

    # get needed dims
    n = div_vel_grad_vel.shape[2:]
    dim = len(ell)

    # get device
    device = div_vel_grad_vel.device

    # make wavenumbers
    k_x = []
    for i, nx in enumerate(n):
        k_x.append(
            torch.cat(
                (
                    torch.arange(start=0, end=nx // 2, step=1, device=device),
                    torch.arange(start=-nx // 2, end=0, step=1, device=device),
                ),
                0,
            ).reshape((i + 2) * [1] + [nx] + (dim - i - 1) * [1])
        )
    # note that for pressure, the zero mode (the mean) cannot be zero for invertability so it is set to 1
    lap = torch.zeros_like(k_x[0])
    for i, k_x_i in enumerate(k_x):
        lap = lap - ((2 * pi / ell[i]) * k_x_i) ** 2
    lap[..., 0, 0] = -1.0

    if p is None:
        # compute fourier transform
        div_vel_grad_vel_h = torch.fft.fftn(
            div_vel_grad_vel, dim=list(range(2, dim + 2))
        )
        div_B_grad_B_h = torch.fft.fftn(div_B_grad_B, dim=list(range(2, dim + 2)))
        ptot_h = (div_B_grad_B_h - rho0 * div_vel_grad_vel_h) / lap
        ptot_h[..., 0, 0] = B2_h[..., 0, 0] / 2.0
    else:
        p_h = torch.fft.fftn(p, dim=list(range(2, dim + 2)))
        ptot_h = p_h + B2_h / 2.0

    # compute laplacian in fourier space
    j = torch.complex(
        torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
    )  # Cuda graphs does not work here
    wx_h = [j * k_x_i * ptot_h * (2 * pi / ell[i]) for i, k_x_i in enumerate(k_x)]

    # inverse fourier transform out
    wx = torch.cat(
        [torch.fft.ifftn(wx_h_i, dim=list(range(2, dim + 2))).real for wx_h_i in wx_h],
        dim=1,
    )
    return wx


def fourier_derivatives_vec_pot(x: Tensor, ell: List[float]) -> List[Tensor]:
    """
    Fourier derivative function for vector potential

    Parameters
    ----------
    x : Tensor
        Input tensor (vector potential)
    ell : List[float]
        Domain lengths

    Returns
    -------
    List[Tensor]
        List containing derivatives, vector potential, magnetic field components and B^2 in Fourier space
    """

    # check that input shape matches domain length
    if len(x.shape) - 2 != len(ell):
        raise ValueError("input shape doesn't match domain dims")

    # set pi from numpy
    pi = float(np.pi)

    # get needed dims
    n = x.shape[2:]
    dim = len(ell)

    # get device
    device = x.device

    # compute fourier transform
    x_h = torch.fft.fftn(x, dim=list(range(2, dim + 2)))

    # make wavenumbers
    k_x = []
    for i, nx in enumerate(n):
        k_x.append(
            torch.cat(
                (
                    torch.arange(start=0, end=nx // 2, step=1, device=device),
                    torch.arange(start=-nx // 2, end=0, step=1, device=device),
                ),
                0,
            ).reshape((i + 2) * [1] + [nx] + (dim - i - 1) * [1])
        )

    # compute laplacian in fourier space
    j = torch.complex(
        torch.tensor([0.0], device=device), torch.tensor([1.0], device=device)
    )  # Cuda graphs does not work here
    Ax_h = j * k_x[0] * x_h * (2 * pi / ell[0])
    Ay_h = j * k_x[1] * x_h * (2 * pi / ell[1])

    B2_h = (Ay_h) ** 2 + (-Ax_h) ** 2

    Bx_h = [j * k_x_i * Ay_h * (2 * pi / ell[i]) for i, k_x_i in enumerate(k_x)]
    By_h = [j * k_x_i * -Ax_h * (2 * pi / ell[i]) for i, k_x_i in enumerate(k_x)]

    # inverse fourier transform out
    wA = torch.cat(
        [
            torch.fft.ifftn(w_h_i, dim=list(range(2, dim + 2))).real
            for w_h_i in [Ax_h, Ay_h]
        ],
        dim=1,
    )
    wB = torch.cat(
        [
            torch.fft.ifftn(w_h_i, dim=list(range(2, dim + 2))).real
            for w_h_i in [Ay_h, -Ax_h]
        ],
        dim=1,
    )
    wx = torch.cat(
        [torch.fft.ifftn(wx_h_i, dim=list(range(2, dim + 2))).real for wx_h_i in Bx_h],
        dim=1,
    )
    wy = torch.cat(
        [torch.fft.ifftn(wx_h_i, dim=list(range(2, dim + 2))).real for wx_h_i in By_h],
        dim=1,
    )

    return wx, wy, wA, wB, B2_h


class FactorizedSpectralConv3d(nn.Module):
    """3D Factorized Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    modes3 : int
        Number of Fourier modes to multiply in third dimension, at most floor(N/2) + 1
    rank : float
        Rank of the decomposition
    factorization : {'CP', 'TT', 'Tucker'}
        Tensor factorization to use to decompose the tensor
    fixed_rank_modes : List[int]
        A list of modes for which the initial value is not modified
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict
        Additional arguments to initialization of factorized tensors
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        rank: float,
        factorization: str,
        fixed_rank_modes: bool,
        decomposition_kwargs: dict,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights2 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights3 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights4 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.reset_parameters()

    def compl_mul3d(self, input: Tensor, weights: Tensor) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        cweights = torch.view_as_complex(weights.to_tensor().contiguous())
        return torch.einsum("bixyz,ioxyz->boxyz", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.normal_(0, self.scale)
        self.weights2.normal_(0, self.scale)
        self.weights3.normal_(0, self.scale)
        self.weights4.normal_(0, self.scale)
