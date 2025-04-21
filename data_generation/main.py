"""Main module for MHD data generation."""

import os
import logging
import math
import multiprocessing as mp
import numpy as np
import torch
from typing import List, Optional, Dict, Any

from data_generation.config import MHDConfig, parse_arguments
from data_generation.random_fields import GRF_Mattern
from data_generation.simulator import MHDSimulator
from data_generation.data_processor import MHDDataProcessor
from data_generation.utils import get_indices_to_process


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("mhd_generation.log")],
    )


def generate_initial_conditions(config: MHDConfig) -> Dict[str, np.ndarray]:
    """
    Generate initial conditions for MHD simulations.

    Args:
        config: MHD configuration

    Returns:
        Dictionary with initial condition arrays
    """
    logger = logging.getLogger("MHDGenerator")

    grf_u = GRF_Mattern(
        dim=2,
        size=config.Nx,
        length=config.Lx,
        nu=config.Nu,
        l=config.l_u,
        sigma=config.sigma_u,
        boundary="periodic",
    )

    grf_A = GRF_Mattern(
        dim=2,
        size=config.Nx,
        length=config.Lx,
        nu=config.Nu,
        l=config.l_A,
        sigma=config.sigma_A,
        boundary="periodic",
    )

    # Generate samples
    logger.info(f"Generating {config.N} samples of initial conditions")
    u0_pot = (
        grf_u.sample(config.N).cpu().numpy().reshape(config.N, config.Nx, config.Ny)
    )
    A0 = grf_A.sample(config.N).cpu().numpy().reshape(config.N, config.Nx, config.Ny)

    return {"u_pot": u0_pot, "A": A0}


def run_simulation(
    sim_idx: int, config: MHDConfig, initial_data: Dict[str, np.ndarray]
) -> None:
    """
    Run a single MHD simulation.

    Args:
        sim_idx: Simulation index
        config: MHD configuration
        initial_data: Initial condition data
    """
    print(f"Starting simulation {sim_idx} on process {mp.current_process().name}")

    # Extract initial conditions for this simulation
    sim_initial_data = {
        "u_pot": initial_data["u_pot"][sim_idx],
        "A": initial_data["A"][sim_idx],
    }

    # Create and run simulator
    simulator = MHDSimulator(config, sim_idx, sim_initial_data)
    simulator.run()


def generate_mhd_data(config: Optional[MHDConfig] = None) -> str:
    """
    Generate MHD simulation data.

    Args:
        config: MHD configuration or None to use command line arguments

    Returns:
        Path to the generated data file
    """
    setup_logging()
    logger = logging.getLogger("MHDGenerator")

    if config is None:
        config = parse_arguments()

    os.makedirs(config.output_dir, exist_ok=True)

    # Expected number of time steps
    Nt = len(np.arange(0, config.tend + config.Dt, config.output_dt)) - 1

    # Get indices of simulations to run
    indices = get_indices_to_process(
        config.N, config.output_dir, Nt, config.skip_exists
    )

    if not indices:
        logger.info("All simulations already completed. Skipping simulation phase.")
    else:
        logger.info(f"Running {len(indices)} simulations out of {config.N} total")

        initial_data = generate_initial_conditions(config)

        # Set up multiprocessing
        try:
            # Use 'fork' method on Unix-like systems (Linux, macOS)
            mp.set_start_method("fork")
        except RuntimeError:
            # If already set, ignore
            pass

        n_workers = max(1, mp.cpu_count() - 1)
        logger.info(f"Using {n_workers} worker processes")

        # Run simulations in parallel
        with mp.Pool(n_workers) as pool:
            args = [(idx, config, initial_data) for idx in indices]
            pool.starmap(run_simulation, args, chunksize=1)

    logger.info("Processing simulation outputs")
    processor = MHDDataProcessor(
        config.output_dir, config.N, config.tend, config.output_dt
    )

    data = processor.process_all_simulations(range(config.N))
    output_file = os.path.join(config.output_dir, "mhd_data_3channel.npy")
    processor.save_combined_data(data)

    logger.info(f"MHD data generation complete. Data saved to {output_file}")
    return output_file


if __name__ == "__main__":
    generate_mhd_data()
