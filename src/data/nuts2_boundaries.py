"""
NUTS-2 boundary utilities.

Download and manage Eurostat NUTS-2 boundary shapefiles for spatial
aggregation and mapping.

Usage
-----
    python -m src.data.nuts2_boundaries
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import requests
from loguru import logger

from src.utils.config import load_config, get_path
from src.utils.constants import NUTS2_CODES, CRS_WGS84

# Eurostat NUTS boundaries download URL (2021 classification, 1:1M scale)
NUTS_BOUNDARIES_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/download/"
    "ref-nuts-2021-01m.shp.zip"
)


def download_nuts_boundaries(
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Download NUTS boundary shapefiles from Eurostat/GISCO.

    Parameters
    ----------
    output_dir : Path
        Directory to save the boundary files.
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    Path
        Path to the extracted shapefile directory.
    """
    zip_path = output_dir / "nuts_boundaries.zip"
    extract_dir = output_dir

    # Check if already downloaded
    shp_files = list(output_dir.glob("*.shp"))
    if shp_files and not overwrite:
        logger.info("NUTS boundaries already downloaded")
        return output_dir

    logger.info("Downloading NUTS-2021 boundaries from Eurostat GISCO...")
    output_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(NUTS_BOUNDARIES_URL, stream=True)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    logger.success(f"NUTS boundaries extracted to {extract_dir}")
    zip_path.unlink()  # Clean up zip

    return extract_dir


def load_italy_nuts2(boundaries_dir: Path) -> gpd.GeoDataFrame:
    """Load Italian NUTS-2 boundaries from the downloaded shapefile.

    Parameters
    ----------
    boundaries_dir : Path
        Directory containing the NUTS boundary shapefiles.

    Returns
    -------
    gpd.GeoDataFrame
        Italian NUTS-2 regions with geometry.
    """
    # Find the NUTS-2 shapefile (may be nested)
    shp_candidates = list(boundaries_dir.rglob("*NUTS_RG*2021*4326*.shp"))
    if not shp_candidates:
        shp_candidates = list(boundaries_dir.rglob("*.shp"))

    if not shp_candidates:
        raise FileNotFoundError(
            f"No shapefiles found in {boundaries_dir}. "
            "Run download_nuts_boundaries() first."
        )

    # Use the first matching shapefile
    shp_path = shp_candidates[0]
    logger.info(f"Loading NUTS boundaries from {shp_path}")

    gdf = gpd.read_file(shp_path)

    # Filter to Italian NUTS-2
    # The column might be NUTS_ID, NUTS_CODE, or similar
    id_col = None
    for candidate in ["NUTS_ID", "NUTS_CODE", "nuts_id", "id"]:
        if candidate in gdf.columns:
            id_col = candidate
            break

    level_col = None
    for candidate in ["LEVL_CODE", "NUTS_LEVEL", "level"]:
        if candidate in gdf.columns:
            level_col = candidate
            break

    if id_col is None:
        raise ValueError(f"Cannot find NUTS ID column. Available: {gdf.columns.tolist()}")

    # Filter to Italy NUTS-2
    italy_mask = gdf[id_col].str.startswith("IT")
    if level_col:
        italy_mask = italy_mask & (gdf[level_col] == 2)

    italy = gdf[italy_mask].copy()
    italy = italy.rename(columns={id_col: "NUTS_ID"})
    italy = italy.to_crs(CRS_WGS84)

    logger.info(f"Loaded {len(italy)} Italian NUTS-2 regions")
    return italy


def setup_boundaries(cfg: dict[str, Any] | None = None) -> gpd.GeoDataFrame:
    """Download (if needed) and load Italian NUTS-2 boundaries.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.

    Returns
    -------
    gpd.GeoDataFrame
        Italian NUTS-2 boundaries.
    """
    if cfg is None:
        cfg = load_config()

    boundaries_dir = get_path(cfg, "raw_data") / "boundaries"
    download_nuts_boundaries(boundaries_dir)
    return load_italy_nuts2(boundaries_dir)


if __name__ == "__main__":
    gdf = setup_boundaries()
    print(gdf[["NUTS_ID"]].to_string())
