"""PDE calculations for MHD simulations with vector potential."""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
from ..utils.fourier_utils import (
    create_wavenumbers,
    compute_derivative,
    compute_laplacian,
    compute_time_derivative,
)


def compute_mhd_pde(
    psi: torch.Tensor,
    A: torch.Tensor,
    Lx: float,
    Ly: float,
    tend: float,
    nu: float,
    eta: float,
    rho0: float,
    p: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PDE residuals for MHD equations using stream function and vector
    potential.

    Args:
        psi: stream function (batch, time, x, y)
        A: z-component of vector potential (batch, time, x, y)
        Lx: Domain length in x direction
        Ly: Domain length in y direction
        tend: Final simulation time
        nu: Kinematic viscosity
        eta: Magnetic diffusivity
        rho0: Fluid density
        p: Pressure field (optional)

    Returns:
        Tuple of PDE residuals (Du, Dv, DA)
    """
    # Get domain dimensions
    batchsize = A.size(0)
    nt = A.size(1)
    nx = A.size(2)
    ny = A.size(3)
    device = A.device
    dt = tend / (nt - 1)

    # Create wavenumbers for spectral derivatives
    kx, ky, kx_first, ky_first = create_wavenumbers(nx, ny, Lx, Ly, device)

    # Compute Laplacian operator for pressure calculation
    lap = -(kx**2 + ky**2)
    lap[..., 0, 0] = -1.0  # Avoid division by zero at zero frequency

    # Compute Fourier transforms
    psi_h = torch.fft.rfftn(psi, dim=[2, 3])
    A_h = torch.fft.rfftn(A, dim=[2, 3])

    # Compute velocity derivatives
    u_h  = -compute_derivative(psi_h, ky_first)
    v_h  = compute_derivative(psi_h, kx_first)
    ux_h = compute_derivative(u_h, kx_first)
    uy_h = compute_derivative(u_h, ky_first)
    vx_h = compute_derivative(v_h, kx_first)
    vy_h = compute_derivative(v_h, ky_first)
    
    # Compute vector potential derivatives
    Ax_h = compute_derivative(A_h, kx_first)
    Ay_h = compute_derivative(A_h, ky_first)

    # Magnetic field components (B = ∇ × A)
    Bx_h = Ay_h  # Bx = ∂A/∂y
    By_h = -Ax_h  # By = -∂A/∂x

    # More derivatives of magnetic field
    Bx_x_h = compute_derivative(Bx_h, kx_first)
    Bx_y_h = compute_derivative(Bx_h, ky_first)
    By_x_h = compute_derivative(By_h, kx_first)
    By_y_h = compute_derivative(By_h, ky_first)

    # Compute Laplacians for diffusion terms
    u_lap_h = compute_laplacian(u_h, kx, ky)
    v_lap_h = compute_laplacian(v_h, kx, ky)
    A_lap_h = compute_laplacian(A_h, kx, ky)

    # Transform all fields back to physical space
    u = torch.fft.irfftn(u_h, dim=[2, 3], s=(nx, ny))
    v = torch.fft.irfftn(v_h, dim=[2, 3], s=(nx, ny))
    ux = torch.fft.irfftn(ux_h, dim=[2, 3], s=(nx, ny))
    uy = torch.fft.irfftn(uy_h, dim=[2, 3], s=(nx, ny))
    vx = torch.fft.irfftn(vx_h, dim=[2, 3], s=(nx, ny))
    vy = torch.fft.irfftn(vy_h, dim=[2, 3], s=(nx, ny))
    Ax = torch.fft.irfftn(Ax_h, dim=[2, 3], s=(nx, ny))
    Ay = torch.fft.irfftn(Ay_h, dim=[2, 3], s=(nx, ny))
    Bx = torch.fft.irfftn(Bx_h, dim=[2, 3], s=(nx, ny))
    By = torch.fft.irfftn(By_h, dim=[2, 3], s=(nx, ny))
    Bx_x = torch.fft.irfftn(Bx_x_h, dim=[2, 3], s=(nx, ny))
    Bx_y = torch.fft.irfftn(Bx_y_h, dim=[2, 3], s=(nx, ny))
    By_x = torch.fft.irfftn(By_x_h, dim=[2, 3], s=(nx, ny))
    By_y = torch.fft.irfftn(By_y_h, dim=[2, 3], s=(nx, ny))
    u_lap = torch.fft.irfftn(u_lap_h, dim=[2, 3], s=(nx, ny))
    v_lap = torch.fft.irfftn(v_lap_h, dim=[2, 3], s=(nx, ny))
    A_lap = torch.fft.irfftn(A_lap_h, dim=[2, 3], s=(nx, ny))
    # u = torch.fft.ifftn(u_h, dim=[-2,-1]).real
    # v = torch.fft.ifftn(v_h, dim=[-2,-1]).real
    # ux = torch.fft.ifftn(ux_h, dim=[-2,-1]).real
    # uy = torch.fft.ifftn(uy_h, dim=[-2,-1]).real
    # vx = torch.fft.ifftn(vx_h, dim=[-2,-1]).real
    # vy = torch.fft.ifftn(vy_h, dim=[-2,-1]).real
    # Ax = torch.fft.ifftn(Ax_h, dim=[-2,-1]).real
    # Ay = torch.fft.ifftn(Ay_h, dim=[-2,-1]).real
    # Bx = torch.fft.ifftn(Bx_h, dim=[-2,-1]).real
    # By = torch.fft.ifftn(By_h, dim=[-2,-1]).real
    # Bx_x = torch.fft.ifftn(Bx_x_h, dim=[-2,-1]).real
    # Bx_y = torch.fft.ifftn(Bx_y_h, dim=[-2,-1]).real
    # By_x = torch.fft.ifftn(By_x_h, dim=[-2,-1]).real
    # By_y = torch.fft.ifftn(By_y_h, dim=[-2,-1]).real
    # u_lap = torch.fft.ifftn(u_lap_h, dim=[-2,-1]).real
    # v_lap = torch.fft.ifftn(v_lap_h, dim=[-2,-1]).real
    # A_lap = torch.fft.ifftn(A_lap_h, dim=[-2,-1]).real

    # omega_h = compute_laplacian(psi_h, kx, ky)
    # j_h = -compute_laplacian(A_h, kx, ky)

    # omega_x_h = compute_derivative(omega_h, kx_first)
    # omega_y_h = compute_derivative(omega_h, ky_first)
    # omega_lap_h = compute_laplacian(omega_h, kx, ky)
    
    # jx_h = compute_derivative(j_h, kx_first)
    # jy_h = compute_derivative(j_h, ky_first)
    # jlap_h = compute_laplacian(j_h, kx, ky)

    # omega = torch.fft.irfftn(omega_h, dim=[2, 3], s=(nx, ny))
    # omega_x = torch.fft.irfftn(omega_x_h, dim=[2, 3], s=(nx, ny))
    # omega_y = torch.fft.irfftn(omega_y_h, dim=[2, 3], s=(nx, ny))
    # omega_lap = torch.fft.irfftn(omega_lap_h, dim=[2, 3], s=(nx, ny))
    
    # j = torch.fft.irfftn(j_h, dim=[2, 3], s=(nx, ny))
    # jx = torch.fft.irfftn(jx_h, dim=[2, 3], s=(nx, ny))
    # jy = torch.fft.irfftn(jy_h, dim=[2, 3], s=(nx, ny))
    # j_lap = torch.fft.irfftn(jlap_h, dim=[2, 3], s=(nx, ny))
    
    B2 = Bx**2 + By**2
    B2_h = torch.fft.rfftn(B2, dim=[2, 3])

    # torch.save(v, "v.pt")
    # torch.save(u, "u.pt")
    # torch.save(vx, "vx.pt")
    # torch.save(uy, "uy.pt")
    # torch.save(By_x, "By_x.pt")
    # torch.save(Bx_y, "Bx_y.pt")
    # torch.save(By_x - Bx_y, "j.pt")
    # torch.save(vx-uy, "omega.pt")
    # print(1/0)
    
    # Calculate pressure
    if p is None:
        # Compute pressure by solving the pressure Poisson equation
        div_vel_grad_vel = ux**2 + 2 * uy * vx + vy**2
        div_B_grad_B = Bx_x**2 + 2 * Bx_y * By_x + By_y**2
        div_vel_grad_vel_h = torch.fft.rfftn(div_vel_grad_vel, dim=[2, 3])
        div_B_grad_B_h = torch.fft.rfftn(div_B_grad_B, dim=[2, 3])

        # Solve Poisson equation for total pressure
        ptot_h = (div_B_grad_B_h - rho0 * div_vel_grad_vel_h) / lap
        ptot_h[..., 0, 0] = B2_h[..., 0, 0] / 2.0  # Set mean pressure

        # Calculate fluid pressure by subtracting magnetic pressure
        p_h = ptot_h - B2_h / 2.0
    else:
        p_h = torch.fft.rfftn(p, dim=[2, 3])
        ptot_h = p_h + B2_h / 2.0

    # Compute pressure gradients
    ptot_x_h = compute_derivative(ptot_h, kx_first)
    ptot_y_h = compute_derivative(ptot_h, ky_first)

    # Transform pressure fields back to physical space
    p = torch.fft.irfftn(p_h, dim=[2, 3], s=(nx, ny))
    ptot = torch.fft.irfftn(ptot_h, dim=[2, 3], s=(nx, ny))
    ptot_x = torch.fft.irfftn(ptot_x_h, dim=[2, 3], s=(nx, ny))
    ptot_y = torch.fft.irfftn(ptot_y_h, dim=[2, 3], s=(nx, ny))
    # p = torch.fft.ifftn(p_h, dim=[-2,-1]).real
    # ptot = torch.fft.ifftn(ptot_h, dim=[-2,-1]).real
    # ptot_x = torch.fft.ifftn(ptot_x_h, dim=[-2,-1]).real
    # ptot_y = torch.fft.ifftn(ptot_y_h, dim=[-2,-1]).real

    # Calculate advection terms
    vel_grad_u = u * ux + v * uy  # (u·∇)u
    vel_grad_v = u * vx + v * vy  # (u·∇)v
    vel_grad_A = u * Ax + v * Ay  # (u·∇)A

    # Calculate Lorentz force terms
    B_grad_Bx = Bx * Bx_x + By * Bx_y  # (B·∇)Bx
    B_grad_By = Bx * By_x + By * By_y  # (B·∇)By

    # Right-hand sides of the MHD equations
    u_rhs = -vel_grad_u - ptot_x / rho0 + B_grad_Bx / rho0 + nu * u_lap
    v_rhs = -vel_grad_v - ptot_y / rho0 + B_grad_By / rho0 + nu * v_lap
    A_rhs = -vel_grad_A + eta * A_lap

    # Compute time derivatives
    u_t = compute_time_derivative(u, dt)
    v_t = compute_time_derivative(v, dt)
    A_t = compute_time_derivative(A, dt)
    # torch.save(A, "A.pt")
    # torch.save(A_t, "A_t.pt")
    # torch.save(vel_grad_A, "vel_grad_A.pt")
    # torch.save(A_lap, "A_lap.pt")
    # print(1/0)

    # Compute PDE residuals (should be zero for exact solution)
    Du = u_t - u_rhs[:, 1:-1]
    Dv = v_t - v_rhs[:, 1:-1]
    DA = A_t - A_rhs[:, 1:-1]

    return Du, Dv, DA, u, v
    # omega_t = compute_time_derivative(omega, dt)
    # Domega = omega_t + (u*omega_x + v*omega_y - (Bx*jx + By*jy)/rho0 - nu * omega_lap)[:, 1:-1]

    # return Domega, DA
    


def compute_pde_loss(
    Du: torch.Tensor,
    Dv: torch.Tensor,
    DA: torch.Tensor,
    Du_weight: float = 1.0,
    Dv_weight: float = 1.0,
    DA_weight: float = 1.0,
    use_weighted_mean: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute PDE loss from residuals.

    Args:
        Du: u-momentum equation residual
        Dv: v-momentum equation residual
        DA: vector potential equation residual
        Du_weight: Weight for u-momentum residual
        Dv_weight: Weight for v-momentum residual
        DA_weight: Weight for vector potential residual
        use_weighted_mean: Whether to use weighted mean for loss calculation

    Returns:
        Total PDE loss and dictionary with individual losses
    """
    # Target zero residual for PDEs
    Du_val = torch.zeros_like(Du)
    Dv_val = torch.zeros_like(Dv)
    DA_val = torch.zeros_like(DA)

    # Compute MSE loss for each PDE residual
    loss_Du = F.mse_loss(Du, Du_val)
    loss_Dv = F.mse_loss(Dv, Dv_val)
    loss_DA = F.mse_loss(DA, DA_val)

    # if u != None:
    #     # print(torch.mean(u**2), torch.mean(Du**2))
    #     loss_Du /= torch.mean(u**2)
    # if v != None:
    #     # print(torch.mean(v**2), torch.mean(Dv**2))
    #     loss_Dv /= torch.mean(v**2)
    # if A != None:
    #     # print(torch.mean(A**2), torch.mean(DA**2))
    #     loss_DA /= torch.mean(A**2)

    # Compute weighted sum
    weight_sum = (Du_weight + Dv_weight + DA_weight) if use_weighted_mean else 1.0
    loss_pde = (
        Du_weight * loss_Du + Dv_weight * loss_Dv + DA_weight * loss_DA
    ) / weight_sum

    # Return total loss and individual components
    return loss_pde, {"Du": loss_Du.item(), "Dv": loss_Dv.item(), "DA": loss_DA.item()}

# def compute_pde_loss(
#     Domega: torch.Tensor,
#     DA: torch.Tensor,
#     Domega_weight: float = 1.0,
#     DA_weight: float = 1.0,
#     use_weighted_mean: bool = False,
# ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     """
#     Compute PDE loss from residuals.

#     Args:
#         Du: u-momentum equation residual
#         Dv: v-momentum equation residual
#         DA: vector potential equation residual
#         Du_weight: Weight for u-momentum residual
#         Dv_weight: Weight for v-momentum residual
#         DA_weight: Weight for vector potential residual
#         use_weighted_mean: Whether to use weighted mean for loss calculation

#     Returns:
#         Total PDE loss and dictionary with individual losses
#     """
#     # Target zero residual for PDEs
#     Domega_val = torch.zeros_like(Domega)
#     DA_val = torch.zeros_like(DA)

#     # Compute MSE loss for each PDE residual
#     loss_Domega = F.mse_loss(Domega, Domega_val)
#     loss_DA = F.mse_loss(DA, DA_val)

#     # if u != None:
#     #     # print(torch.mean(u**2), torch.mean(Du**2))
#     #     loss_Du /= torch.mean(u**2)
#     # if v != None:
#     #     # print(torch.mean(v**2), torch.mean(Dv**2))
#     #     loss_Dv /= torch.mean(v**2)
#     # if A != None:
#     #     # print(torch.mean(A**2), torch.mean(DA**2))
#     #     loss_DA /= torch.mean(A**2)

#     # Compute weighted sum
#     weight_sum = (Domega_weight + DA_weight) if use_weighted_mean else 1.0
#     loss_pde = (
#         Domega_weight * loss_Domega + DA_weight * loss_DA
#     ) / weight_sum

#     # Return total loss and individual components
#     return loss_pde, {"Domega": loss_Domega.item(), "DA": loss_DA.item()}
