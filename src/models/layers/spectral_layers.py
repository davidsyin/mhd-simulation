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


@torch.jit.script
def contract_3D(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixyz,ioxyz->boxyz", a, b)
    return res


class FactorizedSpectralConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes1,
        modes2,
        modes3,
        n_layers=4,
        bias=True,
        scale="auto",
        fft_norm="backward",
        mlp=False,
        rank=0.5,
        factorization="cp",
        fixed_rank_modes=None,
        decomposition_kwargs=dict(),
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.fft_norm = fft_norm
        if mlp:
            raise NotImplementedError()
        else:
            self.mlp = None

        if scale == "auto":
            scale = 1 / (in_channels * out_channels)

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None

        if factorization is not None:
            # Convert fractional rank to an integer if needed
            actual_rank = rank
            if isinstance(rank, float) and 0 < rank < 1:
                # Calculate rank based on tensor shape
                min_dim = min(in_channels, out_channels, modes1, modes2, modes3)
                actual_rank = max(1, int(min_dim * rank))

            self.weight = tltorch.FactorizedTensor.new(
                (
                    4 * n_layers,
                    in_channels,
                    out_channels,
                    self.modes1,
                    self.modes2,
                    self.modes3,
                ),
                rank=actual_rank,  # Use the integer rank
                factorization=factorization,
                dtype=torch.cfloat,
                fixed_rank_modes=fixed_rank_modes,
                **decomposition_kwargs,
            )
            self.weight.normal_(0, scale)
            self._get_weight = self._get_weight_factorized
        else:
            self.weight = nn.Parameter(
                scale
                * torch.randn(
                    4 * n_layers,
                    in_channels,
                    out_channels,
                    self.modes1,
                    self.modes2,
                    self.modes3,
                    dtype=torch.cfloat,
                )
            )
            self._get_weight = self._get_weight_dense

        if bias:
            self.bias = nn.Parameter(scale * torch.randn(self.out_channels, 1, 1, 1))
        else:
            self.bias = 0

    def _get_weight_factorized(self, layer_index, corner_index):
        """Get the weights corresponding to a particular layer,
        corner of the Fourier coefficient (top=0 or bottom=1) -- corresponding to lower frequencies
        and complex_index (real=0 or imaginary=1)
        """
        return (
            self.weight()[4 * layer_index + corner_index, :, :, :, :, :]
            .to_tensor()
            .contiguous()
        )

    def _get_weight_dense(self, layer_index, corner_index):
        """Get the weights corresponding to a particular layer,
        corner of the Fourier coefficient (top=0 or bottom=1) -- corresponding to lower frequencies
        and complex_index (real=0 or imaginary=1)
        """
        return self.weight[4 * layer_index + corner_index, :, :, :, :, :]

    def forward(self, x, indices=0):
        with torch.autocast(device_type="cuda", enabled=False):
            batchsize, channels, height, width, depth = x.shape
            dtype = x.dtype

            # Compute Fourier coeffcients
            x_fft = torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=[2, 3, 4])

            # The output will be of size (batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1)
            out_fft = torch.zeros(
                [batchsize, self.out_channels, height, width, depth // 2 + 1],
                device=x.device,
                dtype=torch.cfloat,
            )
            out_fft[:, :, : self.modes1, : self.modes2, : self.modes3] = contract_3D(
                x_fft[:, :, : self.modes1, : self.modes2, : self.modes3],
                self._get_weight(indices, 0),
            )
            out_fft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = contract_3D(
                x_fft[:, :, -self.modes1 :, : self.modes2, : self.modes3],
                self._get_weight(indices, 1),
            )
            out_fft[:, :, : self.modes1 :, -self.modes2 :, : self.modes3] = contract_3D(
                x_fft[:, :, : self.modes1 :, -self.modes2 :, : self.modes3],
                self._get_weight(indices, 2),
            )
            out_fft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = contract_3D(
                x_fft[:, :, -self.modes1 :, : self.modes2, : self.modes3],
                self._get_weight(indices, 1),
            )
            out_fft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = contract_3D(
                x_fft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3],
                self._get_weight(indices, 3),
            )

            # out_size = (int(height*super_res), int(width*super_res))
            x = torch.fft.irfftn(
                out_fft, s=(height, width, depth), norm=self.fft_norm
            ).type(
                dtype
            )  # (x.size(-2), x.size(-1))) +
            x = x + self.bias

        if self.mlp is not None:
            x = self.mlp(x)

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution
        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single convolution is parametrized, directly use the main class."
            )

        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)
