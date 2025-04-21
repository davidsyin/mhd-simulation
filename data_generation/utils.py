"""Utility functions for MHD data generation."""

import os
import glob
import h5py
import math
import numpy as np
from typing import List, Optional, Tuple, Callable


def compute_autocorrelation(velocity_history: List[float]) -> Optional[np.ndarray]:
    """
    Compute the autocorrelation function r(tau).

    Args:
        velocity_history: List of velocity values over time

    Returns:
        Autocorrelation function or None if not enough data
    """
    velocity_history_array = np.array(velocity_history)
    N = len(velocity_history_array)

    if N < 2:
        return None  # Not enough data

    mean_u = np.mean(velocity_history_array)
    var_u = np.var(velocity_history_array)

    # Avoid division by zero
    if var_u == 0:
        return np.zeros(N)

    autocorr = np.correlate(
        velocity_history_array - mean_u, velocity_history_array - mean_u, mode="full"
    ) / (N * var_u)

    # Keep only positive lags
    autocorr = autocorr[N - 1 :]

    return autocorr


def compute_integral_time_scale(velocity_history: List[float], dt: float) -> float:
    """
    Compute the integral time scale T_int by integrating the autocorrelation function.

    Args:
        velocity_history: List of velocity values over time
        dt: Time step between measurements

    Returns:
        Integral time scale or NaN if computation fails
    """
    autocorr = compute_autocorrelation(velocity_history)

    if autocorr is None:
        return float("nan")

    # Integrate using the trapezoidal rule
    T_int = np.trapz(autocorr, dx=dt)

    return T_int


def check_if_complete(sim_outputs: str, Nt: int = 101) -> bool:
    """
    Check if a simulation has completed by examining output files.

    Args:
        sim_outputs: Glob pattern for simulation output files
        Nt: Expected number of time steps

    Returns:
        True if simulation is complete, False otherwise
    """
    try:
        files = sorted(glob.glob(sim_outputs))
        if not files:
            return False

        file = files[0]
        with h5py.File(file, mode="r") as h5file:
            data_file = h5file["tasks"]
            keys = list(data_file.keys())
            dims = data_file[keys[0]].dims
            t = dims[0]["sim_time"][:]

        return len(t) == Nt
    except Exception:
        return False


def get_indices_to_process(
    n_samples: int, output_dir: str, Nt: int, skip_exists: bool = False
) -> List[int]:
    """
    Get indices of simulations that need to be processed.

    Args:
        n_samples: Total number of samples
        output_dir: Output directory
        Nt: Expected number of time steps
        skip_exists: Whether to skip existing outputs

    Returns:
        List of indices to process
    """
    indices = list(range(n_samples))
    digits = int(math.log10(n_samples)) + 1

    if not skip_exists:
        return indices

    completed_list = []
    for j in range(n_samples):
        sim_output_dir = os.path.join(output_dir, f"output-{j:0{digits}}")
        sim_outputs = os.path.join(sim_output_dir, "*.h5")

        if os.path.exists(sim_output_dir):
            completed = check_if_complete(sim_outputs, Nt=Nt)
        else:
            completed = False

        completed_list.append(completed)

    return [j for j, completed in enumerate(completed_list) if not completed]
