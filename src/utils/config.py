import yaml


def load_config(config_path):
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file

    Returns:
        dict: Loaded configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
