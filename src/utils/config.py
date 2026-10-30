"""Configuration loader and schema validator for VoxGuard."""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

