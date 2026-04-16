"""
Data download script.

Usage
-----
    uv run python scripts/download_data.py --source all
    uv run python scripts/download_data.py --source era5
    uv run python scripts/download_data.py --source boundaries
"""

from __future__ import annotations

import click
from loguru import logger

from src.utils.config import load_config


@click.command()
@click.option(
    "--source",
    type=click.Choice(["all", "era5", "earthmover", "boundaries"]),
    default="all",
    help="Data source to download",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
@click.option("--config", default=None, help="Path to config file")
def main(source: str, overwrite: bool, config: str | None):
    """Download raw data for the project."""
    cfg = load_config(config)

    if source in ("all", "boundaries"):
        logger.info("Downloading NUTS-2 boundaries...")
        from src.data.nuts2_boundaries import setup_boundaries
        setup_boundaries(cfg)

    if source in ("all", "era5"):
        logger.info("Downloading ERA5-Land data (CDS API)...")
        from src.data.download_era5 import download_all_era5
        download_all_era5(cfg, overwrite=overwrite)

    if source == "earthmover":
        logger.info("Downloading ERA5 data via Earthmover (AWS Zarr)...")
        from src.data.download_earthmover import download_all_earthmover
        download_all_earthmover(cfg, overwrite=overwrite)

    logger.success("Download complete!")


if __name__ == "__main__":
    main()
