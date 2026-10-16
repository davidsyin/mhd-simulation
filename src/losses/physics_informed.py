"""Physics-informed loss functions that incorporate domain knowledge constraints."""

from typing import Dict, Any, Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch import Tensor

from .loss_factory import register_loss
from .lp_loss import LpLoss, SpectralLpLoss
from ..physics.constraints import compute_constraints, compute_constraint_loss
from ..physics.pde_solvers import compute_mhd_pde, compute_pde_loss

from ..utils.fourier_utils import create_wavenumbers, compute_derivative

from ..utils import (
    get_dataset_normalizer,
    identify_data_channels,
    apply_denormalization,
)


class MHDVecPotLoss(nn.Module):
    """
    Physics-informed loss function for MHD with stream function and vector
    potential formulation.

    This loss combines data fitting with physical constraints from the MHD equations:
    - Data fitting against ground truth predictions
    - Initial condition matching
    - PDE residual minimization
    - Optional divergence-free constraints derived from stream/vector potentials

    Attributes:
        requires_input_data: Flag indicating this loss requires input data
    """

    # Class flag to indicate this loss requires input data
    requires_input_data: bool = True

    def __init__(
        self,
        nu: float = 1e-4,
        eta: float = 1e-4,
        rho0: float = 1.0,
        data_weight: float = 1.0,
        ic_weight: float = 1.0,
        pde_weight: float = 1.0,
        constraint_weight: float = 1.0,
        use_data_loss: bool = True,
        use_ic_loss: bool = True,
        use_pde_loss: bool = True,
        use_constraint_loss: bool = True,
        psi_weight: float = 1.0,
        A_weight: float = 1.0,
        u_weight: Optional[float] = None,
        v_weight: Optional[float] = None,
        Du_weight: float = 1.0,
        Dv_weight: float = 1.0,
        DA_weight: float = 1.0,
        div_B_weight: float = 1.0,
        div_vel_weight: float = 1.0,
        Lx: float = 1.0,
        Ly: float = 1.0,
        tend: float = 1.0,
        use_weighted_mean: bool = False,
        **kwargs,  # Accept and ignore additional keyword arguments
    ):
        """
        Initialize the MHD vector potential loss function.

        Args:
            nu: Kinematic viscosity
            eta: Magnetic diffusivity
            rho0: Fluid density
            data_weight: Weight for data fitting loss
            ic_weight: Weight for initial condition loss
            pde_weight: Weight for PDE residual loss
            constraint_weight: Weight for constraint loss
            use_data_loss: Whether to include data fitting loss
            use_ic_loss: Whether to include initial condition loss
            use_pde_loss: Whether to include PDE residual loss
            use_constraint_loss: Whether to include constraint loss
            psi_weight: Weight for stream function component
            A_weight: Weight for vector potential
            u_weight: Backward-compatible alias for psi_weight
            v_weight: Backward-compatible ignored velocity weight
            Du_weight: Weight for u-momentum equation residual
            Dv_weight: Weight for v-momentum equation residual
            DA_weight: Weight for vector potential equation residual
            div_B_weight: Weight for magnetic field divergence constraint
            div_vel_weight: Weight for velocity divergence constraint
            Lx: Domain length in x direction
            Ly: Domain length in y direction
            tend: Final simulation time
            use_weighted_mean: Whether to normalize by sum of weights
            **kwargs: Additional keyword arguments (ignored)
        """
        super().__init__()
        self.nu = nu
        self.eta = eta
        self.rho0 = rho0

        # Main loss component weights
        self.data_weight = data_weight
        self.ic_weight = ic_weight
        self.pde_weight = pde_weight
        self.constraint_weight = constraint_weight

        # Loss component flags
        self.use_data_loss = use_data_loss
        self.use_ic_loss = use_ic_loss
        self.use_pde_loss = use_pde_loss
        self.use_constraint_loss = use_constraint_loss

        # Field component weights. Keep u_weight/v_weight accepted so older
        # configs still load while this loss operates on channels [psi, A].
        if u_weight is not None:
            psi_weight = u_weight
        self.psi_weight = psi_weight
        self.A_weight = A_weight

        # PDE residual weights
        self.Du_weight = Du_weight
        self.Dv_weight = Dv_weight
        self.DA_weight = DA_weight

        # Constraint weights
        self.div_B_weight = div_B_weight
        self.div_vel_weight = div_vel_weight

        # Domain parameters
        self.Lx = Lx
        self.Ly = Ly
        self.tend = tend

        # Loss calculation method
        self.use_weighted_mean = use_weighted_mean

    def forward(
        self, dataloader: DataLoader, pred: Tensor, target: Tensor, inputs: Tensor
    ) -> Tensor:
        """
        Calculate the total loss combining data fit and physics constraints.

        Args:
            dataloader: DataLoader containing the dataset (for normalizer access)
            pred: Model predictions with shape [batch, channels, time, x, y]
            target: Ground truth targets with shape [batch, channels, time, x, y]
            inputs: Input data containing initial conditions

        Returns:
            Total weighted loss
        """
        normalizer = get_dataset_normalizer(dataloader)

        # Identify data channels and check for grid embeddings
        data_channel_indices, has_grid_embeddings, _ = identify_data_channels(
            inputs, target
        )

        # Denormalize the data if a normalizer is available
        inputs_denorm, target_denorm, pred_denorm = apply_denormalization(
            inputs, target, pred, normalizer, data_channel_indices, has_grid_embeddings
        )
        if has_grid_embeddings and inputs_denorm.shape[1] != target_denorm.shape[1]:
            inputs_denorm = inputs_denorm[:, data_channel_indices]

        return self.compute_loss(pred_denorm, target_denorm, inputs_denorm)

    def compute_loss(self, pred: Tensor, target: Tensor, inputs: Tensor) -> Tensor:
        """
        Compute weighted loss with all components.

        Args:
            pred: Predicted fields [batch, channels, time, x, y]
            target: Target fields [batch, channels, time, x, y]
            inputs: Input fields [batch, channels, time, x, y]

        Returns:
            Total weighted loss
        """
        # Extract field components
        psi = pred[:, 0]
        A = pred[:, 1]
        
        # Track individual loss components
        loss_components = {}

        # Data loss
        if self.use_data_loss:
            loss_data = self.data_loss(pred, target)
            loss_components["data"] = loss_data.item()
            print("data", loss_data)
        else:
            loss_data = torch.tensor(0.0, device=pred.device)
            loss_components["data"] = 0.0

        # Initial condition loss
        if self.use_ic_loss:
            loss_ic = self.ic_loss(pred, inputs)
            loss_components["ic"] = loss_ic.item()
            print("ic", loss_ic)
        else:
            loss_ic = torch.tensor(0.0, device=pred.device)
            loss_components["ic"] = 0.0

        # PDE loss
        if self.use_pde_loss:
            Du, Dv, DA, u, v = compute_mhd_pde(
                psi, A, self.Lx, self.Ly, self.tend, self.nu, self.eta, self.rho0
            )
            # Du, Dv, DA, u, v = compute_mhd_pde(
            #     target[:,0], target[:,1], self.Lx, self.Ly, self.tend, self.nu, self.eta, self.rho0
            # )
            loss_pde, pde_components = compute_pde_loss(
                Du,
                Dv,
                DA,
                self.Du_weight,
                self.Dv_weight,
                self.DA_weight,
                self.use_weighted_mean,
                u,
                v,
                A,
            )
            print("pde", loss_pde)
            loss_components.update({f"pde_{k}": v for k, v in pde_components.items()})
        else:
            loss_pde = torch.tensor(0.0, device=pred.device)

        # Constraint loss
        if self.use_constraint_loss:
            u, v = self.velocity_from_stream_function(psi)
            div_vel, div_B = compute_constraints(u, v, A, self.Lx, self.Ly, self.tend)
            loss_constraint, constraint_components = compute_constraint_loss(
                div_vel,
                div_B,
                self.div_vel_weight,
                self.div_B_weight,
                self.use_weighted_mean,
            )
            loss_components.update(
                {f"constraint_{k}": v for k, v in constraint_components.items()}
            )
        else:
            loss_constraint = torch.tensor(0.0, device=pred.device)

        # Calculate weight normalization factor
        if self.use_weighted_mean:
            active_weights = (
                (self.data_weight if self.use_data_loss else 0)
                + (self.ic_weight if self.use_ic_loss else 0)
                + (self.pde_weight if self.use_pde_loss else 0)
                + (self.constraint_weight if self.use_constraint_loss else 0)
            )
            weight_sum = max(active_weights, 1.0)  # Avoid division by zero
        else:
            weight_sum = 1.0

        # Compute weighted total loss
        loss = (
            self.data_weight * loss_data
            + self.ic_weight * loss_ic
            + self.pde_weight * loss_pde
            + self.constraint_weight * loss_constraint
        ) / weight_sum

        # Store total loss
        loss_components["total"] = loss.item()

        return loss

    def compute_loss(self, pred: Tensor, target: Tensor, inputs: Tensor) -> Tensor:
        """
        Compute weighted loss with all components.

        Args:
            pred: Predicted fields [batch, channels, time, x, y]
            target: Target fields [batch, channels, time, x, y]
            inputs: Input fields [batch, channels, time, x, y]

        Returns:
            Total weighted loss
        """
        # Extract field components
        psi, A = pred[:, 0], pred[:, 1]

        # Track individual loss components
        loss_components = {}

        # Data loss
        if self.use_data_loss:
            loss_data = self.data_loss(pred, target)
            loss_components["data"] = loss_data.item()
            print("data", loss_data)
            # torch.save(pred, "pred.pt")
            # torch.save(target, "target.pt")
            # print(1/0)
        else:
            loss_data = torch.tensor(0.0, device=pred.device)
            loss_components["data"] = 0.0

        # Initial condition loss
        if self.use_ic_loss:
            loss_ic = self.ic_loss(pred, inputs)
            loss_components["ic"] = loss_ic.item()
            print("ic", loss_ic)
        else:
            loss_ic = torch.tensor(0.0, device=pred.device)
            loss_components["ic"] = 0.0

        # PDE loss
        if self.use_pde_loss:
            Du, Dv, DA, u, v = compute_mhd_pde(
                psi, A, self.Lx, self.Ly, self.tend, self.nu, self.eta, self.rho0
            )
            # Du_target, Dv_target, DA_target, _, _ = compute_mhd_pde(
            #     target[:,0], target[:,1], self.Lx, self.Ly, self.tend, self.nu, self.eta, self.rho0
            # )
            # loss_pde_target, _ = compute_pde_loss(
            #     Du_target,
            #     Dv_target,
            #     DA_target,
            #     self.Du_weight,
            #     self.DA_weight,
            #     self.use_weighted_mean,
            # )
            # print("pde target", loss_pde_target)
            loss_pde, pde_components = compute_pde_loss(
                Du,
                Dv,
                DA,
                self.Du_weight,
                self.DA_weight,
                self.use_weighted_mean,
            )
            print("pde", loss_pde)
            loss_components.update({f"pde_{k}": v for k, v in pde_components.items()})
        else:
            loss_pde = torch.tensor(0.0, device=pred.device)

        # Constraint loss
        if self.use_constraint_loss:
            u, v = self.velocity_from_stream_function(psi)
            div_vel, div_B = compute_constraints(u, v, A, self.Lx, self.Ly, self.tend)
            loss_constraint, constraint_components = compute_constraint_loss(
                div_vel,
                div_B,
                self.div_vel_weight,
                self.div_B_weight,
                self.use_weighted_mean,
            )
            loss_components.update(
                {f"constraint_{k}": v for k, v in constraint_components.items()}
            )
        else:
            loss_constraint = torch.tensor(0.0, device=pred.device)

        # Calculate weight normalization factor
        if self.use_weighted_mean:
            active_weights = (
                (self.data_weight if self.use_data_loss else 0)
                + (self.ic_weight if self.use_ic_loss else 0)
                + (self.pde_weight if self.use_pde_loss else 0)
                + (self.constraint_weight if self.use_constraint_loss else 0)
            )
            weight_sum = max(active_weights, 1.0)  # Avoid division by zero
        else:
            weight_sum = 1.0

        # Compute weighted total loss
        loss = (
            self.data_weight * loss_data
            + self.ic_weight * loss_ic
            + self.pde_weight * loss_pde
            + self.constraint_weight * loss_constraint
        ) / weight_sum

        # Store total loss
        loss_components["total"] = loss.item()

        return loss

    def pot_to_vec(self, pot, kx_first, ky_first, psi=False):
        h = torch.fft.fftn(torch.tensor(pot), dim=[2,3])
        u = torch.fft.ifftn(compute_derivative(h, ky_first))
        v = -torch.fft.ifftn(compute_derivative(h, kx_first))
        if psi:
            u = -u
            v = -v
        return u, v

    def data_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute data fitting loss using Lp loss.

        Args:
            pred: Predicted fields [batch, channels, time, x, y]
            target: Target fields [batch, channels, time, x, y]

        Returns:
            Data fitting loss
        """
        lploss = SpectralLpLoss(size_average=True, spec_min=0.0, spec_max=0.1, weight=0.5)
        lploss_normal = SpectralLpLoss(size_average=True, spec_min=0.0, spec_max=1.0, weight=0.0)
        psi, A = target[:, 0], target[:, 1]

        # Extract field components
        psi_pred, A_pred = pred[:, 0], pred[:, 1]
        # _, _, kx_first, ky_first = create_wavenumbers(128,128,1,1,"cuda")
        
        # u_pred, v_pred = self.pot_to_vec(psi_pred, kx_first, ky_first, psi=True)
        # u, v = self.pot_to_vec(psi, kx_first, ky_first, psi=True)

        # # Bx_pred, By_pred = self.pot_to_vec(A_pred, kx_first, ky_first, psi=False)
        # # Bx, By = self.pot_to_vec(A, kx_first, ky_first, psi=False)

        # u_loss = lploss(u_pred, u)
        # v_loss = lploss(v_pred, v)
        # # Bx_loss = lploss(Bx_pred, Bx)
        # # By_loss = lploss(By_pred, By)
        # A_loss = lploss(A_pred, A)
        # print(u_loss, v_loss, A_loss)
        # loss_data = ((u_loss + v_loss) / 2 + A_loss)

        # # print(torch.mean(psi_target), torch.mean(A_target), torch.std(psi_target), torch.std(A_target))
        # # Compute component losses
        loss_psi = lploss_normal(psi_pred, psi)
        loss_A = lploss_normal(A_pred, A)
        print("psi", loss_psi)
        print("A", loss_A)

        # Apply component weights
        if self.use_weighted_mean:
            weight_sum = self.psi_weight + self.A_weight
        else:
            weight_sum = 1.0

        # # Compute weighted loss
        loss_data = (self.psi_weight * loss_psi + self.A_weight * loss_A) / weight_sum

        return loss_data

    def ic_loss(self, pred: Tensor, inputs: Tensor) -> Tensor:
        """
        Compute initial condition loss using Lp loss.

        Args:
            pred: Predicted fields [batch, channels, time, x, y]
            inputs: Input fields with initial conditions [batch, channels, time, x, y]

        Returns:
            Initial condition loss
        """
        lploss = LpLoss(size_average=True)
        # print("pred", pred.shape)
        # print("input", inputs.shape)

        # Extract initial conditions (t=0)
        ic_pred = pred[:, :, 0]
        ic_target = inputs[:, :, 0]
 
        # Extract field components at t=0
        psi_ic_pred, A_ic_pred = ic_pred[:, 0], ic_pred[:, 1]
        psi_ic_target, A_ic_target = ic_target[:, 0], ic_target[:, 1]

        # Compute component losses
        loss_psi_ic = lploss(psi_ic_pred, psi_ic_target)
        loss_A_ic = lploss(A_ic_pred, A_ic_target)

        # Apply component weights
        if self.use_weighted_mean:
            weight_sum = self.psi_weight + self.A_weight
        else:
            weight_sum = 1.0

        # Compute weighted loss
        loss_ic = (
            self.psi_weight * loss_psi_ic + self.A_weight * loss_A_ic
        ) / weight_sum

        return loss_ic

    def velocity_from_stream_function(self, psi: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Reconstruct velocity from stream function: u = -dpsi/dy, v = dpsi/dx.

        Args:
            psi: Stream function with shape [batch, time, x, y]

        Returns:
            Tuple of velocity components (u, v)
        """
        from ..utils.fourier_utils import create_wavenumbers, compute_derivative

        nx = psi.size(2)
        ny = psi.size(3)
        k_x, k_y = create_wavenumbers(nx, ny, self.Lx, self.Ly, psi.device)

        psi_h = torch.fft.fftn(psi, dim=[2, 3])
        u_h = -compute_derivative(psi_h, k_y)
        v_h = compute_derivative(psi_h, k_x)

        u = torch.fft.irfftn(u_h, dim=[2, 3], s=(nx, ny))
        v = torch.fft.irfftn(v_h, dim=[2, 3], s=(nx, ny))

        return u, v


@register_loss("physics-informed")
def create_physics_informed_loss(params: Dict[str, Any]) -> MHDVecPotLoss:
    """
    Create a physics-informed loss function for MHD with vector potential formulation.

    Args:
        params: Dictionary of parameters for the loss function

    Returns:
        Configured MHDVecPotLoss instance
    """
    # You can optionally filter out the 'type' key here if you want
    # params_copy = {k: v for k, v in params.items() if k != 'type'}
    # return MHDVecPotLoss(**params_copy)

    # Or simply pass all params which is fine since we added **kwargs to the constructor
    return MHDVecPotLoss(**params)
