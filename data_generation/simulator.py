"""MHD simulator using Dedalus framework."""

import os
import h5py
import logging
import numpy as np
import dedalus.public as d3
from typing import List, Dict, Optional, Tuple, Any, Callable

from data_generation.config import MHDConfig
from data_generation.utils import compute_integral_time_scale


class MHDSimulator:
    """
    Magnetohydrodynamics simulator using Dedalus.

    This class encapsulates the setup and execution of 2D MHD simulations
    with initial conditions derived from random fields.
    """

    def __init__(
        self, config: MHDConfig, sim_index: int, initial_data: Dict[str, np.ndarray]
    ):
        """
        Initialize the MHD simulator.

        Args:
            config: Configuration parameters
            sim_index: Index of this simulation
            initial_data: Initial condition data (velocity potential and vector potential)
        """
        self.config = config
        self.sim_index = sim_index
        self.initial_data = initial_data
        self.logger = logging.getLogger(f"MHDSimulator_{sim_index}")

        # Setup output paths
        digits = int(np.log10(config.N)) + 1 if config.N > 1 else 1
        self.output_dir = os.path.join(
            config.output_dir, f"output-{sim_index:0{digits}}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Storage for velocity history (for autocorrelation)
        self.velocity_history: List[float] = []

        # Setup the problem
        self._setup_problem()

    def _setup_problem(self) -> None:
        """Set up the MHD problem using Dedalus."""
        # Coordinates and bases
        coords = d3.CartesianCoordinates("x", "y")
        self.dist = d3.Distributor(coords, dtype=self.config.dtype)

        self.xbasis = d3.RealFourier(
            coords["x"],
            size=self.config.Nx,
            bounds=(0, self.config.Lx),
            dealias=self.config.dealias,
        )
        self.ybasis = d3.RealFourier(
            coords["y"],
            size=self.config.Ny,
            bounds=(0, self.config.Ly),
            dealias=self.config.dealias,
        )

        # Fields
        self._create_fields(coords)

        # Simulation set-up
        self._setup_equations(coords)  # Substitutions and equations

        self.solver = self.problem.build_solver(self.config.timestepper)
        self.solver.stop_sim_time = self.config.tend + self.config.Dt

        self._set_initial_conditions()

        self._setup_analysis()

        self._setup_cfl()

        self._setup_flow_properties()

    def _create_fields(self, coords) -> None:
        """Create the Dedalus fields for the simulation."""
        # Fields
        self.p = self.dist.Field(name="p", bases=(self.xbasis, self.ybasis))
        self.s = self.dist.Field(name="s", bases=(self.xbasis, self.ybasis))
        self.u = self.dist.VectorField(
            coords, name="u", bases=(self.xbasis, self.ybasis)
        )
        self.B = self.dist.VectorField(
            coords, name="B", bases=(self.xbasis, self.ybasis)
        )
        self.A = self.dist.Field(name="A", bases=(self.xbasis, self.ybasis))
        self.B2 = self.dist.Field(name="B2", bases=(self.xbasis, self.ybasis))
        self.u_pot = self.dist.Field(name="u_pot", bases=(self.xbasis, self.ybasis))
        self.Bx = self.dist.Field(name="Bx", bases=(self.xbasis, self.ybasis))
        self.By = self.dist.Field(name="By", bases=(self.xbasis, self.ybasis))
        self.ux = self.dist.Field(name="ux", bases=(self.xbasis, self.ybasis))
        self.uy = self.dist.Field(name="uy", bases=(self.xbasis, self.ybasis))
        self.tau_p = self.dist.Field(name="tau_p")

    def _setup_equations(self, coords) -> None:
        """Set up the equations for the MHD problem."""
        # Unit vectors
        ex, ey = coords.unit_vector_fields(self.dist)

        # Differential operators
        curl2d_scalar = lambda x: -d3.skew(d3.grad(x))
        curl2d_vector = lambda x: -d3.div(d3.skew(x))

        # Physical quantities and relations
        self.B = curl2d_scalar(self.A)
        self.B2 = d3.dot(self.B, self.B)
        self.Bx = self.B @ ex
        self.By = self.B @ ey
        self.ux = self.u @ ex
        self.uy = self.u @ ey

        # Problem definition
        self.problem = d3.IVP(
            [self.u, self.p, self.A, self.tau_p, self.s], namespace=locals()
        )

        # Add equations
        nu = self.config.nu
        eta = self.config.eta
        D = self.config.D
        rho0 = self.config.rho0

        self.problem.add_equation(
            f"dt(u) + grad(p)/rho0 - {nu}*lap(u) = - 0.5*grad(B2)/rho0 - u@grad(u) + B@grad(B)"
        )
        self.problem.add_equation(f"dt(s) - {D}*lap(s) = - u@grad(s)")
        self.problem.add_equation(f"dt(A) - {eta}*lap(A) = - u@grad(A)")
        self.problem.add_equation("div(u) + tau_p = 0")
        self.problem.add_equation("integ(p) = 0")  # Pressure gauge

    def _set_initial_conditions(self) -> None:
        """Set initial conditions from random fields."""
        # Set velocity potential and initial velocity field
        self.u_pot["g"] = self.initial_data["u_pot"]
        u0 = self._curl2d_scalar(self.u_pot).evaluate()
        u0.change_scales(1)
        self.u["g"] = u0["g"]

        # Set scalar and vector potential
        self.s["g"] = self.initial_data["u_pot"]  # Using same field for scalar
        self.A["g"] = self.initial_data["A"]

    def _curl2d_scalar(self, scalar_field):
        """Compute 2D curl of a scalar field."""
        return -d3.skew(d3.grad(scalar_field))

    def _setup_analysis(self) -> None:
        """Set up data analysis and output."""
        self.snapshots = self.solver.evaluator.add_file_handler(
            self.output_dir,
            sim_dt=self.config.output_dt,
            max_writes=self.config.max_writes,
        )

        # Save required fields
        self.snapshots.add_task(self.ux, name="u_x")
        self.snapshots.add_task(self.uy, name="u_y")
        self.snapshots.add_task(self.A, name="vector potential")

    def _setup_cfl(self) -> None:
        """Set up CFL condition for adaptive time stepping."""
        self.CFL = d3.CFL(
            self.solver,
            initial_dt=self.config.max_timestep,
            cadence=10,
            safety=0.2,
            threshold=0.1,
            max_change=1.5,
            min_change=0.5,
            max_dt=self.config.max_timestep,
        )
        self.CFL.add_velocity(self.u)

    def _setup_flow_properties(self) -> None:
        """Set up flow property tracking."""
        self.flow = d3.GlobalFlowProperty(self.solver, cadence=10)
        self.flow.add_property(d3.dot(self.u, self.u), name="w2")
        self.flow.add_property(d3.dot(self.B, self.B), name="B2")
        self.flow.add_property(d3.div(self.B), name="divB")

    def run(self) -> None:
        """Run the simulation."""
        # Open a log file for this simulation
        log_file = os.path.join(self.output_dir, "integral_time_scale.csv")
        with open(log_file, "w") as f:
            f.write("iteration,sim_time,T_int\n")

        self.logger.info(f"Starting simulation {self.sim_index} in {self.output_dir}")

        try:
            # Main loop
            while self.solver.proceed:
                # Determine timestep
                if self.config.use_cfl:
                    timestep = self.CFL.compute_timestep()
                else:
                    timestep = self.config.Dt

                # Step the simulation
                self.solver.step(timestep)

                # Store velocity data for autocorrelation
                self.velocity_history.append(np.mean(self.solver.state[0].data))

                if (self.solver.iteration) % self.config.log_iter == 0:
                    max_w = np.sqrt(self.flow.max("w2"))
                    max_B = np.sqrt(self.flow.max("B2"))
                    max_divB = self.flow.max("divB")

                    # Compute integral time scale
                    T_int = compute_integral_time_scale(self.velocity_history, timestep)

                    self.logger.info(
                        f"Iteration={self.solver.iteration}, "
                        f"Time={self.solver.sim_time:.3g}, "
                        f"dt={timestep:.3g}, "
                        f"T_int={T_int:.3g}, "
                        f"max(w)={max_w:.3g}, "
                        f"max(B)={max_B:.3g}, "
                        f"max(divB)={max_divB:.3g}"
                    )

                    with open(log_file, "a") as f:
                        f.write(
                            f"{self.solver.iteration},{self.solver.sim_time:.6g},{T_int:.6g}\n"
                        )

            self.logger.info(f"Finished simulation {self.sim_index}")

        except Exception as e:
            self.logger.error(f"Exception in simulation {self.sim_index}: {e}")
            raise
        finally:
            self.solver.log_stats()
