"""
Download ERA5-Land reanalysis data from the Copernicus Climate Data Store.

Downloads daily 2m temperature and 2m dewpoint temperature for Italy at
hourly resolution (to derive daily Tmax, Tmin, Tmean) for all summer months
(June–September) across the study period (2012–2022).

Usage
-----
    python -m src.data.download_era5
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cdsapi
from loguru import logger

from src.utils.config import load_config, get_path, get_cds_api_key


def build_era5_request(
    year: int,
    months: list[int],
    variables: list[str],
    bbox: dict[str, float],
) -> dict[str, Any]:
    """Build a CDS API request dictionary for ERA5-Land.

    Parameters
    ----------
    year : int
        Year to download.
    months : list of int
        Months to include (e.g., [6, 7, 8, 9]).
    variables : list of str
        ERA5-Land variable names.
    bbox : dict
        Bounding box with keys: north, south, west, east.

    Returns
    -------
    dict
        CDS API request parameters.
    """
    return {
        "product_type": "reanalysis",
        "variable": variables,
        "year": str(year),
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": [bbox["north"], bbox["west"], bbox["south"], bbox["east"]],
        "format": "netcdf",
    }


def download_era5_year(
    year: int,
    cfg: dict[str, Any],
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Download ERA5-Land data for a single year.

    Parameters
    ----------
    year : int
        Year to download.
    cfg : dict
        Project configuration dictionary.
    output_dir : Path
        Directory to save the downloaded file.
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    Path
        Path to the downloaded NetCDF file.
    """
    output_file = output_dir / f"era5land_italy_{year}.nc"

    if output_file.exists() and not overwrite:
        logger.info(f"ERA5-Land {year} already exists, skipping: {output_file}")
        return output_file

    temp_cfg = cfg["temperature"]
    study_cfg = cfg["study"]

    request = build_era5_request(
        year=year,
        months=study_cfg["summer_months"],
        variables=temp_cfg["variables"],
        bbox=temp_cfg["bbox"],
    )

    logger.info(f"Downloading ERA5-Land data for {year}...")
    client = cdsapi.Client()

    client.retrieve(
        "reanalysis-era5-land",
        request,
        str(output_file),
    )

    logger.success(f"Downloaded ERA5-Land {year} → {output_file}")
    return output_file


def download_all_era5(
    cfg: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download ERA5-Land data for all years in the study period.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary. If None, loads from default path.
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    list of Path
        Paths to all downloaded NetCDF files.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = get_path(cfg, "raw_data") / "climate"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = cfg["study"]["start_year"]
    end = cfg["study"]["end_year"]
    files = []

    for year in range(start, end + 1):
        try:
            f = download_era5_year(year, cfg, output_dir, overwrite)
            files.append(f)
        except Exception as e:
            logger.error(f"Failed to download ERA5-Land {year}: {e}")

    return files


if __name__ == "__main__":
    download_all_era5()
