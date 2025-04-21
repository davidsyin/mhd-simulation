"""
Random field generation for MHD initial conditions.
Implements Gaussian Random Field with Matérn covariance function.
"""

import torch
from math import pi, gamma, sqrt
from typing import Literal, Optional, Tuple, Union, List


class GRF_Mattern:
    """
    Gaussian Random Field generator with Matérn covariance function.

    This class generates random fields with spatial correlation using
    the Matérn covariance function, suitable for initial conditions in
    fluid dynamics simulations.
    """

    def __init__(
        self,
        dim: int,
        size: int,
        length: float = 1.0,
        nu: Optional[float] = None,
        l: float = 0.1,
        sigma: float = 1.0,
        boundary: Literal["periodic", "dirichlet", "neumann"] = "periodic",
        constant_eig: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Gaussian Random Field generator.

        Args:
            dim: Dimension of the field (1D, 2D, or 3D)
            size: Number of points in each dimension
            length: Physical length of the domain
            nu: Smoothness parameter (if None, uses RBF kernel)
            l: Correlation length
            sigma: Standard deviation of the field
            boundary: Boundary conditions ("periodic", "dirichlet", or "neumann")
            constant_eig: Value for the constant eigenvalue (mean of the field)
            device: PyTorch device (CPU or GPU)
        """
        self.dim = dim
        self.device = device
        self.bc = boundary
        self.size = tuple([size] * dim)

        # Calculate coefficients based on the Matérn covariance function
        if self.bc == "dirichlet":
            constant_eig = None

        # Set up parameters for the covariance function
        if nu is not None:
            kappa = sqrt(2 * nu) / l
            alpha = nu + 0.5 * dim
            self.eta2 = (
                size**dim
                * sigma
                * (4.0 * pi) ** (0.5 * dim)
                * gamma(alpha)
                / (kappa**dim * gamma(nu))
            )
        else:
            self.eta2 = size**dim * sigma * (sqrt(2.0 * pi) * l) ** dim

        # Wavenumber setup
        k_max = size // 2
        if self.bc == "periodic":
            const = (4.0 * (pi**2)) / (length**2)
        else:
            const = (pi**2) / (length**2)

        # Set up spectral representation based on dimension
        if dim == 1:
            self._setup_1d_spectrum(
                k_max, const, nu, kappa, length, l, alpha, constant_eig
            )
        elif dim == 2:
            self._setup_2d_spectrum(
                size, k_max, const, nu, kappa, length, l, alpha, constant_eig
            )
        elif dim == 3:
            self._setup_3d_spectrum(
                size, k_max, const, nu, kappa, length, l, alpha, constant_eig
            )
        else:
            raise ValueError(f"Dimension {dim} not supported. Must be 1, 2, or 3.")

    def _setup_1d_spectrum(
        self,
        k_max: int,
        const: float,
        nu: Optional[float],
        kappa: Optional[float],
        length: float,
        l: float,
        alpha: Optional[float],
        constant_eig: Optional[float],
    ) -> None:
        """Set up 1D spectral representation."""
        k = torch.cat(
            (
                torch.arange(start=0, end=k_max, step=1, device=self.device),
                torch.arange(start=-k_max, end=0, step=1, device=self.device),
            ),
            0,
        )

        k2 = k**2
        if nu is not None:
            eigs = 1.0 + (const / (kappa * length) ** 2 * k2)
            self.sqrt_eig = self.eta2 / (length**self.dim) * eigs ** (-alpha / 2.0)
        else:
            self.sqrt_eig = (
                self.eta2
                / (length**self.dim)
                * torch.exp(-((l) ** 2) * const * k2 / 4.0)
            )

        # Set constant eigenvalue (for mean field)
        if constant_eig is not None:
            self.sqrt_eig[0] = constant_eig
        else:
            self.sqrt_eig[0] = 0.0

    def _setup_2d_spectrum(
        self,
        size: int,
        k_max: int,
        const: float,
        nu: Optional[float],
        kappa: Optional[float],
        length: float,
        l: float,
        alpha: Optional[float],
        constant_eig: Optional[float],
    ) -> None:
        """Set up 2D spectral representation."""
        wavenumbers = torch.cat(
            (
                torch.arange(start=0, end=k_max, step=1, device=self.device),
                torch.arange(start=-k_max, end=0, step=1, device=self.device),
            ),
            0,
        ).repeat(size, 1)

        k_x = wavenumbers.transpose(0, 1)
        k_y = wavenumbers

        k2 = k_x**2 + k_y**2
        if nu is not None:
            eigs = 1.0 + (const / (kappa * length) ** 2 * k2)
            self.sqrt_eig = self.eta2 / (length**self.dim) * eigs ** (-alpha / 2.0)
        else:
            self.sqrt_eig = (
                self.eta2
                / (length**self.dim)
                * torch.exp(-((l) ** 2) * const * k2 / 4.0)
            )

        # Set constant eigenvalue (for mean field)
        if constant_eig is not None:
            self.sqrt_eig[0, 0] = constant_eig
        else:
            self.sqrt_eig[0, 0] = 0.0

    def _setup_3d_spectrum(
        self,
        size: int,
        k_max: int,
        const: float,
        nu: Optional[float],
        kappa: Optional[float],
        length: float,
        l: float,
        alpha: Optional[float],
        constant_eig: Optional[float],
    ) -> None:
        """Set up 3D spectral representation."""
        wavenumbers = torch.cat(
            (
                torch.arange(start=0, end=k_max, step=1, device=self.device),
                torch.arange(start=-k_max, end=0, step=1, device=self.device),
            ),
            0,
        ).repeat(size, size, 1)

        k_x = wavenumbers.transpose(1, 2)
        k_y = wavenumbers
        k_z = wavenumbers.transpose(0, 2)

        k2 = k_x**2 + k_y**2 + k_z**2
        if nu is not None:
            eigs = 1.0 + (const / (kappa * length) ** 2 * k2)
            self.sqrt_eig = self.eta2 / (length**self.dim) * eigs ** (-alpha / 2.0)
        else:
            self.sqrt_eig = (
                self.eta2
                / (length**self.dim)
                * torch.exp(-((l) ** 2) * const * k2 / 4.0)
            )

        # Set constant eigenvalue (for mean field)
        if constant_eig is not None:
            self.sqrt_eig[0, 0, 0] = constant_eig
        else:
            self.sqrt_eig[0, 0, 0] = 0.0

    def sample(self, N: int) -> torch.Tensor:
        """
        Generate samples from the random field.

        Args:
            N: Number of samples to generate

        Returns:
            Tensor of shape (N, size, size, ...)
        """
        coeff = torch.randn(N, *self.size, dtype=torch.cfloat, device=self.device)

        # Apply boundary condition constraints
        if self.bc == "dirichlet":
            coeff.real[:] = 0
        if self.bc == "neumann":
            coeff.imag[:] = 0

        # Apply spectral coefficient scaling
        coeff = self.sqrt_eig * coeff

        # Transform to real space
        u = torch.fft.irfftn(coeff, self.size, norm="backward")
        return u
