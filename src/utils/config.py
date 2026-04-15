"""Configuration loader for the project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Project root is two levels up from this file (src/utils/config.py → Project/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load the project configuration from YAML.

    Parameters
    ----------
    config_path : Path or str, optional
        Path to the config file. Defaults to ``config/config.yaml``
        relative to the project root.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a relative path from the config to an absolute path.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (output of :func:`load_config`).
    key : str
        Key under ``cfg["paths"]``, e.g. ``"raw_data"``.

    Returns
    -------
    Path
        Absolute path resolved relative to the project root.
    """
    return PROJECT_ROOT / cfg["paths"][key]


def get_cds_api_key(cfg: dict[str, Any]) -> str:
    """Get the CDS API key from environment or config.

    Priority: environment variable ``CDS_API_KEY`` > config file.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    str
        CDS API key.

    Raises
    ------
    ValueError
        If no API key is found.
    """
    key = os.environ.get("CDS_API_KEY")
    if key:
        return key
    key = cfg.get("cds_api", {}).get("key")
    if key and key != "YOUR_CDS_API_KEY":
        return key
    raise ValueError(
        "CDS API key not found. Set the CDS_API_KEY environment variable "
        "or add it to config/config.yaml under cds_api.key"
    )


def get_region_mapping(cfg: dict[str, Any]) -> dict[str, str]:
    """Return the NUTS-2 code → region name mapping.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.

    Returns
    -------
    dict
        Mapping of NUTS-2 codes to region names.
    """
    return cfg["regions"]["nuts2_codes"]
