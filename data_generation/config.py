"""Configuration module for MHD data generation."""

import argparse
import numpy as np
from dataclasses import dataclass
from typing import Any, Optional, Type, Union
import dedalus.public as d3


@dataclass
class MHDConfig:
    """Configuration class for MHD simulation."""
    
    # Domain parameters
    Lx: float = 1.0
    Ly: float = 1.0
    Nx: int = 128
    Ny: int = 128
    
    # Physical parameters
    Re: float = 5e2
    ReM: float = 5e2
    Schmidt: float = 1.0
    rho0: float = 1.0
    
    # Simulation parameters
    dealias: float = 3/2
    tend: float = 1.0
    Dt: float = 1e-3
    timestepper: Any = d3.RK443
    max_timestep: float = 1e-2
    output_dt: float = 1e-2
    log_iter: int = 10
    dtype: Type = np.float64
    max_writes: Optional[int] = None
    use_cfl: bool = False
    
    # Output parameters
    output_dir: str = "mhd_data"
    movie_dir: str = "movie"
    
    # Initial condition parameters
    L: float = 1.0
    N: int = 1  # Number of simulations
    l_u: float = 0.1  # Length scale for velocity potential
    l_A: float = 0.1  # Length scale for magnetic vector potential
    Nu: Optional[float] = None  # Smoothness parameter
    sigma_u: float = 0.1  # Amplitude for velocity potential
    sigma_A: float = 5e-3  # Amplitude for magnetic vector potential
    skip_exists: bool = False
    
    # Analysis options
    calculate_time_scale: bool = False  # Whether to calculate integral time scale
    
    @property
    def nu(self) -> float:
        """Get kinematic viscosity."""
        return 1 / self.Re
    
    @property
    def eta(self) -> float:
        """Get magnetic diffusivity."""
        return 1 / self.ReM
    
    @property
    def D(self) -> float:
        """Get diffusion coefficient."""
        return self.nu / self.Schmidt


def parse_arguments() -> MHDConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(description="MHD data generation for NODs")
    
    # Domain parameters
    parser.add_argument("--Lx", type=float, default=1.0, help="Length of domain in x direction")
    parser.add_argument("--Ly", type=float, default=1.0, help="Length of domain in y direction")
    parser.add_argument("--Nx", type=int, default=128, help="Number of points in x direction")
    parser.add_argument("--Ny", type=int, default=128, help="Number of points in y direction")
    
    # Physical parameters
    parser.add_argument("--Re", type=float, default=5e2, help="Reynolds number")
    parser.add_argument("--ReM", type=float, default=5e2, help="Magnetic Reynolds number")
    parser.add_argument("--Schmidt", type=float, default=1.0, help="Schmidt number")
    parser.add_argument("--rho0", type=float, default=1.0, help="Density of fluid")
    
    # Simulation parameters
    parser.add_argument("-d", "--dealias", type=float, default=3/2, help="Dealiasing factor")
    parser.add_argument("-T", "--tend", type=float, default=1.0, help="End time of simulation")
    parser.add_argument("-t", "--Dt", type=float, default=1e-3, help="Timestep size")
    parser.add_argument("--timestepper", default=d3.RK443, help="Timestepper type")
    parser.add_argument("--max_timestep", type=float, default=1e-2, help="Maximum timestep for CFL control")
    parser.add_argument("--output_dt", type=float, default=1e-2, help="Time between outputs")
    parser.add_argument("--log_iter", type=int, default=10, help="Iterations between logging")
    parser.add_argument("--dtype", default=np.float64, help="Datatype for simulation")
    parser.add_argument("--use_cfl", action="store_true", help="Use timestep computed based on CFL")
    
    # Output parameters
    parser.add_argument("-o", "--output_dir", type=str, default="mhd_data", help="Directory to store outputs")
    parser.add_argument("-m", "--movie_dir", type=str, default="movie", help="Directory to store movie")
    parser.add_argument("-w", "--max_writes", type=int, default=None, help="Maximum file writes")
    
    # Initial condition parameters
    parser.add_argument("-N", "--N", type=int, default=1, help="Number of simulations to run")
    parser.add_argument("-L", "--L", type=float, default=1.0, help="Length of domain for generating data")
    parser.add_argument("--l_u", type=float, default=0.1, help="Length scale for velocity potential")
    parser.add_argument("--l_A", type=float, default=0.1, help="Length scale for magnetic vector potential")
    parser.add_argument("--Nu", type=float, default=None, help="Smoothness parameter for GRF")
    parser.add_argument("--sigma_u", type=float, default=0.1, help="Amplitude of velocity potential")
    parser.add_argument("--sigma_A", type=float, default=5e-3, help="Amplitude of magnetic vector potential")
    parser.add_argument("-s", "--skip_exists", action="store_true", help="Skip existing output files")
    
    # Analysis options
    parser.add_argument("--calculate_time_scale", action="store_true", help="Calculate and report integral time scale")
    
    args = parser.parse_args()
    
    # Create and return config
    return MHDConfig(
        Lx=args.Lx,
        Ly=args.Ly,
        Nx=args.Nx,
        Ny=args.Ny,
        Re=args.Re,
        ReM=args.ReM,
        Schmidt=args.Schmidt,
        rho0=args.rho0,
        dealias=args.dealias,
        tend=args.tend,
        Dt=args.Dt,
        timestepper=args.timestepper,
        max_timestep=args.max_timestep,
        output_dt=args.output_dt,
        log_iter=args.log_iter,
        dtype=args.dtype,
        max_writes=args.max_writes,
        use_cfl=args.use_cfl,
        output_dir=args.output_dir,
        movie_dir=args.movie_dir,
        L=args.L,
        N=args.N,
        l_u=args.l_u,
        l_A=args.l_A,
        Nu=args.Nu,
        sigma_u=args.sigma_u,
        sigma_A=args.sigma_A,
        skip_exists=args.skip_exists,
        calculate_time_scale=args.calculate_time_scale
    )
