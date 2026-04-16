"""
Earthmover ERA5 (AWS Zarr) data acquisition.

Provides high-performance, cloud-native access to ERA5 surface variables
stashed on AWS S3 via the Arraylake platform.

Usage
-----
    python -m src.data.download_earthmover
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr
from arraylake import Client
from loguru import logger
from tqdm import tqdm

from src.utils.config import load_config, get_path


def download_earthmover_year(
    year: int,
    cfg: dict[str, Any],
    output_dir: Path,
    overwrite: bool = False,
) -> Path | None:
    """Download a regional subset for a specific year from Earthmover.

    Parameters
    ----------
    year : int
        Year to download.
    cfg : dict
        Project configuration.
    output_dir : Path
        Directory to save the resulting NetCDF.
    overwrite : bool, optional
        Whether to overwrite existing files.

    Returns
    -------
    Path or None
        Path to the saved file, or None if failed.
    """
    output_path = output_dir / f"era5_earthmover_italy_{year}.nc"

    if output_path.exists() and not overwrite:
        logger.info(f"Skipping {year} — already exists: {output_path.name}")
        return output_path

    try:
        client = Client()
        repo_name = cfg["earthmover"]["repo"]
        branch = cfg["earthmover"]["branch"]
        group = cfg["earthmover"]["group"]

        logger.info(f"Connecting to Earthmover repo: {repo_name}...")
        repo = client.get_repo(repo_name)
        session = repo.readonly_session(branch=branch)
        ds = xr.open_zarr(session.store, group=group)

        # Subset time: June 1 to Sept 30
        start_date = f"{year}-06-01"
        end_date = f"{year}-09-30"
        
        # Subset space: Italy bounding box
        bbox = cfg["temperature"]["bbox"]
        
        # Earthmover/Zarr indexing might be different (lat: 90 to -90, lon: 0 to 360)
        # Note: ERA5 longitudes are 0-360. West 6.5, East 19.0 are fine.
        
        logger.info(f"Subsetting {year} (Summer)...")
        # Use .sel with slices. Handle latitude descending order.
        subset = ds.sel(
            time=slice(start_date, f"{year}-09-30 23:00:00"),
            latitude=slice(bbox["north"], bbox["south"]),
            longitude=slice(bbox["west"], bbox["east"])
        )

        # Select only the variables we need
        subset = subset[["t2", "d2"]]

        # Rename to match CDS standard (t2m, d2m) for downstream consistency
        mapping = cfg["earthmover"].get("variable_mapping", {})
        if mapping:
            subset = subset.rename(mapping)

        logger.info(f"Loading {year} into memory...")
        subset = subset.load()

        # Fix time encoding: the Zarr store inherits 'hours since 1975-01-01'
        # encoding which overflows on NetCDF write. Strip it and let xarray
        # re-encode cleanly.
        for var in subset.variables:
            subset[var].encoding.clear()

        # Set explicit, clean encoding for numeric variables
        encoding = {}
        for var in subset.data_vars:
            encoding[var] = {"dtype": "float32", "zlib": True, "complevel": 4}
        encoding["time"] = {"units": "hours since 2000-01-01", "calendar": "proleptic_gregorian"}

        logger.info(f"Saving {year} to NetCDF...")
        subset.to_netcdf(output_path, encoding=encoding)
        
        logger.success(f"Successfully saved {year} → {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to download {year} from Earthmover: {e}")
        return None


def download_all_earthmover(
    cfg: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download all years defined in the study period.

    Parameters
    ----------
    cfg : dict, optional
        Configuration.
    overwrite : bool, optional
        Overwrite existing.

    Returns
    -------
    list of Path
        Paths to downloaded files.
    """
    if cfg is None:
        cfg = load_config()

    output_dir = get_path(cfg, "raw_data") / "climate"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_year = cfg["study"]["start_year"]
    end_year = cfg["study"]["end_year"]
    years = range(start_year, end_year + 1)

    downloaded = []
    logger.info(f"Starting Earthmover bulk download for {len(years)} years...")

    for year in tqdm(years, desc="Downloading ERA5 (Earthmover)"):
        path = download_earthmover_year(year, cfg, output_dir, overwrite=overwrite)
        if path:
            downloaded.append(path)

    logger.success(f"Earthmover bulk download complete. Total files: {len(downloaded)}")
    return downloaded


if __name__ == "__main__":
    download_all_earthmover()
