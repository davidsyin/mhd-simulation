import os
import re
import torch
import numpy as np


def extract_reynolds_number(filename):
    """Extract Reynolds number from config filename"""
    match = re.search(r"Re(\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def setup_device(gpu_index):
    """Setup and return device"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(
            f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    return device


def find_config_files(config_dir, prefix, reynolds_numbers=None):
    """
    Find config files matching prefix and optional Reynolds numbers

    Args:
        config_dir: Directory to search for config files
        prefix: Prefix of config files (e.g., "config_tfno3d_Re")
        reynolds_numbers: List of Reynolds numbers to filter by (optional)

    Returns:
        list: Sorted list of config file paths
    """
    config_files = []

    if reynolds_numbers and isinstance(reynolds_numbers, str):
        reynolds_numbers = [int(re.strip()) for re in reynolds_numbers.split(",")]

    # Find matching config files
    for file in os.listdir(config_dir):
        if file.startswith(prefix) and file.endswith(".yaml"):
            re_number = extract_reynolds_number(file)

            # Skip files that don't match requested Reynolds numbers
            if reynolds_numbers and re_number not in reynolds_numbers:
                continue

            config_files.append(os.path.join(config_dir, file))

    return sorted(
        config_files, key=lambda x: extract_reynolds_number(os.path.basename(x))
    )


def check_path_exists(path, is_directory=False, error_msg=None):
    """Check if a path exists and exit with error message if not"""
    if is_directory:
        if not os.path.isdir(path):
            print(error_msg or f"Directory not found: {path}")
            return False
    else:
        if not os.path.exists(path):
            print(error_msg or f"File not found: {path}")
            return False

    return True
