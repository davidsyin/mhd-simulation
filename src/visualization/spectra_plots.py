"""Energy spectra visualization for MHD simulations."""

import os
from typing import Union, Tuple, List, Dict
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..utils import (
    get_dataset_normalizer,
    identify_data_channels,
    apply_denormalization,
)


def compute_energy_spectra(
    data: Union[torch.Tensor, np.ndarray], dx: float = 1.0, dy: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute energy spectra from 2D data using FFT.

    Args:
        data: Tensor of shape (batch_size, channels, height, width) or (channels, height, width)
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction

    Returns:
        k: Wave numbers
        spectra_kin: Kinetic energy spectra (u_x, u_y components)
        spectra_mag: Magnetic energy spectra (A component)
    """
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    # Handle different input shapes
    if len(data.shape) == 3:  # (channels, height, width)
        data = data[np.newaxis, ...]  # Add batch dimension

    batch_size, channels, height, width = data.shape

    # Assume channel 0,1 are velocity components (u_x, u_y) and channel 2 is magnetic potential (A)
    # Check if we have the right number of channels
    assert channels >= 3, "Data must have at least 3 channels (u_x, u_y, A)"

    # Compute wave numbers
    kx = 2 * np.pi * np.fft.fftfreq(width, dx)
    ky = 2 * np.pi * np.fft.fftfreq(height, dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)

    k_max = np.sqrt(np.max(kx**2) + np.max(ky**2))
    k_bins = np.linspace(0, k_max, min(width, height) // 2)

    spectra_kin = np.zeros((batch_size, len(k_bins) - 1))
    spectra_mag = np.zeros((batch_size, len(k_bins) - 1))

    for b in range(batch_size):
        u_x_fft = np.fft.fft2(data[b, 0])
        u_y_fft = np.fft.fft2(data[b, 1])

        energy_kin = 0.5 * (np.abs(u_x_fft) ** 2 + np.abs(u_y_fft) ** 2)

        a_fft = np.fft.fft2(data[b, 2])

        # Compute energy from magnetic potential
        # For MHD, we approximate magnetic energy using gradients of magnetic potential
        # Bx = dA/dy, By = -dA/dx, so |B|^2 = |∇A|^2 = |k|^2 |A|^2 in Fourier space
        energy_mag = k_grid**2 * np.abs(a_fft) ** 2

        # Compute 1D spectra by binning
        for i in range(len(k_bins) - 1):
            mask = (k_grid >= k_bins[i]) & (k_grid < k_bins[i + 1])
            if np.any(mask):
                spectra_kin[b, i] = np.mean(energy_kin[mask])
                spectra_mag[b, i] = np.mean(energy_mag[mask])

    # Use bin centers for k values
    k = 0.5 * (k_bins[:-1] + k_bins[1:])

    return k, spectra_kin, spectra_mag


def save_spectra_plots(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    output_dir: str,
    num_samples: int = 2,
    use_log_scale: bool = True,
) -> None:
    """Save energy spectra plots for the best model.

    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        device: Device to run model on
        epoch: Current epoch
        output_dir: Directory to save plots
        num_samples: Number of samples to plot
        use_log_scale: Whether to use log scale for plotting
    """
    model.eval()

    best_spectra_dir = os.path.join(output_dir, "best_model_spectra")
    os.makedirs(best_spectra_dir, exist_ok=True)

    normalizer = get_dataset_normalizer(val_loader)

    with torch.no_grad():
        try:
            x, y = next(iter(val_loader))
            x, y = x.to(device), y.to(device)

            data_channel_indices, has_grid_embeddings, _ = identify_data_channels(x, y)

            pred = model(x)

            _, y_denorm, pred_denorm = apply_denormalization(
                x, y, pred, normalizer, data_channel_indices, has_grid_embeddings
            )

            # Calculate grid spacing (assuming uniform grid)
            dx, dy = 1.0, 1.0  # Default values
            if has_grid_embeddings:
                try:
                    # Extract grid coordinates
                    grid_x = x[0, 1].unique()  # Assuming 2nd channel is x coordinates
                    grid_y = x[0, 2].unique()  # Assuming 3rd channel is y coordinates

                    # Calculate grid spacing
                    dx = float((grid_x[1] - grid_x[0]).cpu().numpy())
                    dy = float((grid_y[1] - grid_y[0]).cpu().numpy())
                except (IndexError, RuntimeError) as e:
                    print(f"Warning: Could not extract grid spacing: {e}")
                    print("Using default grid spacing dx=dy=1.0")

            # # Debug print
            # print(f"Using grid spacing dx={dx}, dy={dy} for spectra calculation")

            # Store all computed spectra data
            all_spectra_data = []

            # Process samples
            for i in range(min(num_samples, x.shape[0])):
                # Compute energy spectra for both true and predicted data
                # Get last time step for both
                y_last = y_denorm[i, :, -1].cpu()
                pred_last = pred_denorm[i, :, -1].cpu()

                # Calculate spectra
                k, true_spectra_kin, true_spectra_mag = compute_energy_spectra(
                    y_last, dx, dy
                )
                _, pred_spectra_kin, pred_spectra_mag = compute_energy_spectra(
                    pred_last, dx, dy
                )

                xmin = None
                for idx, (e_k_true, e_m_true) in enumerate(
                    zip(true_spectra_kin[0], true_spectra_mag[0])
                ):
                    if e_k_true > 1e-8 or e_m_true > 1e-8:
                        xmin = max(k[idx] * 0.9, k[0])
                        break
                if xmin is None or (use_log_scale and xmin <= 0):
                    xmin = max(0.1, k[0])

                # Find min/max energy values for y-axis
                min_energy = float("inf")
                max_energy = 0
                for spectra in [
                    true_spectra_kin[0],
                    true_spectra_mag[0],
                    pred_spectra_kin[0],
                    pred_spectra_mag[0],
                ]:
                    for val in spectra:
                        if val > 0 and val < min_energy:
                            min_energy = val
                        if val > max_energy:
                            max_energy = val

                ymin = min_energy * 0.5 if min_energy != float("inf") else 1e-10
                if use_log_scale and ymin <= 0:
                    ymin = 1e-10

                ymax = max_energy * 1.1

                plt.figure(figsize=(8, 6))

                # Plot with appropriate scale
                if use_log_scale:
                    plt.loglog(k, true_spectra_kin[0], "k-", label="$E_{kin}$ True")
                    plt.loglog(k, pred_spectra_kin[0], "b-", label="$E_{kin}$ Pred")
                    plt.loglog(k, true_spectra_mag[0], "k--", label="$E_{mag}$ True")
                    plt.loglog(k, pred_spectra_mag[0], "b--", label="$E_{mag}$ Pred")
                else:
                    plt.plot(k, true_spectra_kin[0], "k-", label="$E_{kin}$ True")
                    plt.plot(k, pred_spectra_kin[0], "b-", label="$E_{kin}$ Pred")
                    plt.plot(k, true_spectra_mag[0], "k--", label="$E_{mag}$ True")
                    plt.plot(k, pred_spectra_mag[0], "b--", label="$E_{mag}$ Pred")

                plt.xlabel("k")
                plt.ylabel("E(k)")
                plt.title(
                    f"Energy Spectra - Best Model (Epoch {epoch}), Sample {i + 1}"
                )

                plt.xlim(left=xmin, right=k[-1])
                plt.ylim(bottom=ymin, top=ymax)

                plt.legend(loc="upper right")
                plt.tight_layout()

                fig_path = os.path.join(best_spectra_dir, f"spectra_sample_{i}.png")
                plt.savefig(fig_path, bbox_inches="tight")
                plt.close()

                print(
                    f"Energy spectra plot for best model (epoch {epoch}), sample {i + 1} saved to {fig_path}"
                )

                # Save data for this sample
                sample_data = {
                    "k": k,
                    "true_spectra_kin": true_spectra_kin[0],
                    "true_spectra_mag": true_spectra_mag[0],
                    "pred_spectra_kin": pred_spectra_kin[0],
                    "pred_spectra_mag": pred_spectra_mag[0],
                    "sample_index": i,
                    "epoch": epoch,
                    "dx": dx,
                    "dy": dy,
                }
                all_spectra_data.append(sample_data)

                # Also save individual NumPy files
                np_file_path = os.path.join(best_spectra_dir, f"spectra_sample_{i}.npz")
                np.savez(
                    np_file_path,
                    k=k,
                    true_spectra_kin=true_spectra_kin[0],
                    true_spectra_mag=true_spectra_mag[0],
                    pred_spectra_kin=pred_spectra_kin[0],
                    pred_spectra_mag=pred_spectra_mag[0],
                    metadata=np.array(
                        [{"sample_index": i, "epoch": epoch, "dx": dx, "dy": dy}],
                        dtype=object,
                    ),
                )

            # Save all samples to a pickle file
            import pickle

            pickle_file_path = os.path.join(best_spectra_dir, "best_model_spectra.pkl")
            with open(pickle_file_path, "wb") as f:
                pickle.dump(all_spectra_data, f)
            print(f"Saved all best model spectra data to {pickle_file_path}\n")

        except Exception as e:
            print(f"Warning: Could not create spectra plots for best model: {e}")
            import traceback

            traceback.print_exc()
            return


def load_best_spectra_data(output_dir: str) -> List[Dict]:
    """Load the saved energy spectra data for the best model.

    Args:
        output_dir: Directory where best model spectra data is saved

    Returns:
        List of dictionaries containing spectra data for each sample
    """
    import pickle

    best_spectra_dir = os.path.join(output_dir, "best_model_spectra")
    pickle_file_path = os.path.join(best_spectra_dir, "best_model_spectra.pkl")

    if not os.path.exists(pickle_file_path):
        print(f"Warning: No saved spectra data found at {pickle_file_path}")
        return []

    with open(pickle_file_path, "rb") as f:
        spectra_data = pickle.load(f)

    return spectra_data
