"""
Process ERA5-Land climate data: spatial aggregation to NUTS-2 regions.

Takes raw ERA5-Land NetCDF files (hourly, gridded) and produces a daily
regional climate dataset with Tmax, Tmin, Tmean, and dewpoint temperature
for each Italian NUTS-2 region.

Usage
-----
    python -m src.data.process_climate
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from src.utils.config import load_config, get_path
from src.utils.constants import KELVIN_OFFSET, NUTS2_CODES
from src.utils.io import save_dataframe


def load_nuts2_boundaries(boundaries_path: Path) -> gpd.GeoDataFrame:
    """Load NUTS-2 boundaries for Italy.

    Loads the 20 project NUTS-2 regions. Because ISTAT reports
    Trentino-Alto Adige as a single region, the Eurostat polygons for
    ITH1 (Bolzano) and ITH2 (Trento) are dissolved into one "ITH1" geometry
    so that the climate spatial average covers the full combined region.

    Parameters
    ----------
    boundaries_path : Path
        Path to the NUTS-2 shapefile or GeoJSON.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with NUTS-2 geometries for Italy (20 regions).
    """
    gdf = gpd.read_file(boundaries_path)
    # Load project regions plus ITH2 (needed for the ITH1 merge below)
    load_codes = list(NUTS2_CODES) + ["ITH2"]
    italy = gdf[gdf["NUTS_ID"].isin(load_codes)].copy()
    italy = italy.to_crs("EPSG:4326")

    # Merge ITH1 (Bolzano) and ITH2 (Trento) into a single ITH1 polygon
    # to match ISTAT's combined Trentino-Alto Adige regional reporting.
    trentino = italy[italy["NUTS_ID"].isin(["ITH1", "ITH2"])]
    merged_geom = trentino.geometry.unary_union
    ith1_idx = italy.index[italy["NUTS_ID"] == "ITH1"][0]
    italy.at[ith1_idx, "geometry"] = merged_geom
    italy.at[ith1_idx, "NUTS_NAME"] = "Trentino-Alto Adige/Südtirol"
    italy = italy[italy["NUTS_ID"] != "ITH2"].copy()

    logger.info(f"Loaded {len(italy)} Italian NUTS-2 boundaries (ITH1+ITH2 dissolved into ITH1)")
    return italy


def compute_daily_stats(ds: xr.Dataset) -> xr.Dataset:
    """Compute daily Tmax, Tmin, Tmean from hourly ERA5-Land data.

    Parameters
    ----------
    ds : xr.Dataset
        Hourly ERA5-Land dataset with ``t2m`` and ``d2m`` variables.

    Returns
    -------
    xr.Dataset
        Daily dataset with tmax, tmin, tmean, d2m_mean (all in °C).
    """
    # Convert from Kelvin to Celsius
    t2m_c = ds["t2m"] - KELVIN_OFFSET
    d2m_c = ds["d2m"] - KELVIN_OFFSET

    daily = xr.Dataset(
        {
            "tmax": t2m_c.resample(valid_time="1D").max(),
            "tmin": t2m_c.resample(valid_time="1D").min(),
            "tmean": t2m_c.resample(valid_time="1D").mean(),
            "dewpoint_mean": d2m_c.resample(valid_time="1D").mean(),
        }
    )
    return daily


def spatial_average_to_nuts2(
    daily_ds: xr.Dataset,
    nuts2_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Area-weighted average of gridded data to NUTS-2 regions.

    Uses a simple mask-based approach: for each NUTS-2 polygon, identifies
    grid cells whose centers fall within the polygon and averages them.

    Parameters
    ----------
    daily_ds : xr.Dataset
        Daily gridded climate data (lat/lon).
    nuts2_gdf : gpd.GeoDataFrame
        NUTS-2 boundaries.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, nuts2_code, tmax, tmin, tmean, dewpoint_mean.
    """
    records = []

    # Determine lat/lon coordinate names
    lat_name = "latitude" if "latitude" in daily_ds.coords else "lat"
    lon_name = "longitude" if "longitude" in daily_ds.coords else "lon"

    lats = daily_ds[lat_name].values
    lons = daily_ds[lon_name].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    from shapely.geometry import Point

    for _, row in nuts2_gdf.iterrows():
        nuts2_code = row["NUTS_ID"]
        polygon = row.geometry

        # Find grid cells inside the polygon
        mask = np.zeros_like(lat_grid, dtype=bool)
        for i in range(lat_grid.shape[0]):
            for j in range(lat_grid.shape[1]):
                if polygon.contains(Point(lon_grid[i, j], lat_grid[i, j])):
                    mask[i, j] = True

        if not mask.any():
            logger.warning(f"No grid cells found for {nuts2_code}, skipping")
            continue

        # Compute spatial mean for each time step and variable
        time_coord = "valid_time" if "valid_time" in daily_ds.coords else "time"
        times = daily_ds[time_coord].values
        for t_idx, t_val in enumerate(times):
            rec = {
                "date": pd.Timestamp(t_val),
                "nuts2_code": nuts2_code,
            }
            for var in ["tmax", "tmin", "tmean", "dewpoint_mean"]:
                if var in daily_ds:
                    data = daily_ds[var].values[t_idx]  # shape: (lat, lon)
                    rec[var] = float(np.nanmean(data[mask]))
            records.append(rec)

        logger.info(f"Processed region {nuts2_code}: {mask.sum()} grid cells")

    df = pd.DataFrame(records)
    return df


def process_climate_data(
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Full pipeline: load ERA5-Land NetCDFs → daily regional climate data.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Daily climate data for all NUTS-2 regions and years.
    """
    if cfg is None:
        cfg = load_config()

    raw_dir = get_path(cfg, "raw_data") / "climate"
    boundaries_path = get_path(cfg, "raw_data") / "boundaries" / "NUTS_RG_01M_2021_4326.shp"
    output_path = get_path(cfg, "interim_data") / "daily_regional_climate.parquet"

    # Load boundaries
    nuts2_gdf = load_nuts2_boundaries(boundaries_path)

    # Process each year
    all_frames = []
    start = cfg["study"]["start_year"]
    end = cfg["study"]["end_year"]

    for year in range(start, end + 1):
        # Support both standard CDS and Earthmover filename patterns
        nc_file = raw_dir / f"era5land_italy_{year}.nc"
        if not nc_file.exists():
             nc_file = raw_dir / f"era5_earthmover_italy_{year}.nc"
            
        if not nc_file.exists():
            logger.warning(f"Missing ERA5-Land file for {year}: {nc_file}")
            continue

        logger.info(f"Processing ERA5-Land {year} (Source: {nc_file.name})...")
        ds = xr.open_dataset(nc_file)
        
        # Ensure standard coordinate names
        if "time" in ds.coords and "valid_time" not in ds.coords:
            ds = ds.rename({"time": "valid_time"})
            
        daily = compute_daily_stats(ds)
        regional = spatial_average_to_nuts2(daily, nuts2_gdf)
        all_frames.append(regional)
        ds.close()

    if not all_frames:
        raise FileNotFoundError("No ERA5-Land files found. Run download_era5 first.")

    df = pd.concat(all_frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["nuts2_code", "date"]).reset_index(drop=True)

    save_dataframe(df, output_path, index=False)
    return df


if __name__ == "__main__":
    process_climate_data()
