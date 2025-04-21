"""Process and convert MHD simulation data."""

import os
import glob
import h5py
import numpy as np
from typing import List, Dict, Optional


class MHDDataProcessor:
    """Process MHD simulation data from HDF5 files to numpy arrays."""

    def __init__(
        self, output_dir: str, n_sims: int, stop_time: float, output_dt: float
    ):
        """
        Initialize the data processor.

        Args:
            output_dir: Directory containing simulation outputs
            n_sims: Number of simulations
            stop_time: Final simulation time
            output_dt: Time between outputs
        """
        self.output_dir = output_dir
        self.n_sims = n_sims
        self.stop_time = stop_time
        self.output_dt = output_dt
        self.num_timesteps = int(stop_time / output_dt) + 1
        self.digits = int(np.log10(n_sims)) + 1 if n_sims > 1 else 1

    def process_simulation(
        self, sim_idx: int, output_array: np.ndarray, data_idx: int
    ) -> None:
        """
        Process data from a single simulation.

        Args:
            sim_idx: Simulation index
            output_array: Array to store processed data
            data_idx: Index in the output array for this simulation
        """
        sim_output_dir = os.path.join(
            self.output_dir, f"output-{sim_idx:0{self.digits}}"
        )
        files = sorted(glob.glob(os.path.join(sim_output_dir, "*.h5")))

        if not files:
            print(f"Warning: No HDF5 files found in {sim_output_dir}")
            return

        # Read data from h5 files
        for file in files:
            with h5py.File(file, "r") as f:
                # Get the time axis length
                tmax = f["tasks"]["u_x"].shape[0]

                # Copy data to the output array
                for t in range(min(tmax, output_array.shape[1])):
                    output_array[data_idx, t, :, :, 0] = f["tasks"]["u_x"][t, ...]  # ux
                    output_array[data_idx, t, :, :, 1] = f["tasks"]["u_y"][t, ...]  # uy
                    output_array[data_idx, t, :, :, 2] = f["tasks"]["vector potential"][
                        t, ...
                    ]  # A

    def process_all_simulations(self, indices: List[int]) -> np.ndarray:
        """
        Process data from all simulations and return a combined array.

        Args:
            indices: List of simulation indices to process

        Returns:
            Array containing processed data from all simulations
        """
        # Create the output array
        data = np.zeros((len(indices), self.num_timesteps, 128, 128, 3))

        # Process each simulation
        for idx, sim_idx in enumerate(indices):
            self.process_simulation(sim_idx, data, idx)

        return data

    def save_combined_data(
        self, data: np.ndarray, filename: str = "mhd_data_3channel.npy"
    ) -> None:
        """
        Save the combined data to a numpy file.

        Args:
            data: Array containing processed data
            filename: Output filename
        """
        output_path = os.path.join(self.output_dir, filename)
        np.save(output_path, data)
        print(f"Saved combined data to {output_path}")
