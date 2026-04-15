"""
Process socioeconomic data for RSVI construction.

Loads and standardizes socioeconomic indicators from ISTAT and Eurostat
(poverty rates, GDP per capita, age structure, population density) at the
NUTS-2 regional level.

Usage
-----
    python -m src.data.process_socioeconomic
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.utils.config import load_config, get_path, get_region_mapping
from src.utils.constants import NUTS2_CODES
from src.utils.io import save_dataframe


def load_age_structure(filepath: Path) -> pd.DataFrame:
    """Load ISTAT age structure data (% population by age group).

    Parameters
    ----------
    filepath : Path
        Path to the age structure CSV/Excel file.

    Returns
    -------
    pd.DataFrame
        With columns: nuts2_code, year, pct_pop_65plus, pct_pop_75plus, pct_pop_80plus.
    """
    logger.info(f"Loading age structure data from {filepath}")
    df = pd.read_csv(filepath) if filepath.suffix == ".csv" else pd.read_excel(filepath)
    logger.info(f"Loaded {len(df)} age structure records")
    return df


def load_economic_indicators(filepath: Path) -> pd.DataFrame:
    """Load economic indicators (GDP per capita, poverty rate, disposable income).

    Parameters
    ----------
    filepath : Path
        Path to the economic indicators file.

    Returns
    -------
    pd.DataFrame
        With columns: nuts2_code, year, gdp_per_capita, poverty_rate_absolute,
        disposable_income.
    """
    logger.info(f"Loading economic indicators from {filepath}")
    df = pd.read_csv(filepath) if filepath.suffix == ".csv" else pd.read_excel(filepath)
    logger.info(f"Loaded {len(df)} economic records")
    return df


def load_urban_density(filepath: Path) -> pd.DataFrame:
    """Load urban density / population density data.

    Parameters
    ----------
    filepath : Path
        Path to the population density file.

    Returns
    -------
    pd.DataFrame
        With columns: nuts2_code, year, population_density, urbanization_rate.
    """
    logger.info(f"Loading urban density data from {filepath}")
    df = pd.read_csv(filepath) if filepath.suffix == ".csv" else pd.read_excel(filepath)
    logger.info(f"Loaded {len(df)} urban density records")
    return df


def merge_socioeconomic_indicators(
    age_df: pd.DataFrame,
    econ_df: pd.DataFrame,
    urban_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all socioeconomic indicators into a single panel.

    Parameters
    ----------
    age_df : pd.DataFrame
        Age structure data.
    econ_df : pd.DataFrame
        Economic indicators.
    urban_df : pd.DataFrame
        Urban density data.

    Returns
    -------
    pd.DataFrame
        Merged socioeconomic panel with all indicators.
    """
    merge_keys = ["nuts2_code", "year"]

    merged = age_df.copy()

    if not econ_df.empty:
        merged = merged.merge(econ_df, on=merge_keys, how="outer")

    if not urban_df.empty:
        merged = merged.merge(urban_df, on=merge_keys, how="outer")

    # Invert GDP and income so higher values = higher vulnerability
    if "gdp_per_capita" in merged.columns:
        merged["gdp_per_capita_inv"] = -merged["gdp_per_capita"]
    if "disposable_income" in merged.columns:
        merged["disposable_income_inv"] = -merged["disposable_income"]

    logger.info(
        f"Merged socioeconomic panel: {len(merged)} rows, "
        f"{len(merged.columns)} columns"
    )
    return merged


def process_socioeconomic_data(
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Full pipeline: load and merge all socioeconomic indicators.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Processed socioeconomic panel data.
    """
    if cfg is None:
        cfg = load_config()

    raw_dir = get_path(cfg, "raw_data") / "socioeconomic"
    output_path = get_path(cfg, "interim_data") / "socioeconomic_processed.parquet"

    # Look for data files (flexible naming)
    age_files = list(raw_dir.glob("*age*")) + list(raw_dir.glob("*demograph*"))
    econ_files = list(raw_dir.glob("*econ*")) + list(raw_dir.glob("*gdp*")) + list(raw_dir.glob("*poverty*"))
    urban_files = list(raw_dir.glob("*urban*")) + list(raw_dir.glob("*density*"))

    age_df = pd.DataFrame()
    econ_df = pd.DataFrame()
    urban_df = pd.DataFrame()

    if age_files:
        age_df = load_age_structure(age_files[0])
    else:
        logger.warning("No age structure data found in data/raw/socioeconomic/")

    if econ_files:
        econ_df = load_economic_indicators(econ_files[0])
    else:
        logger.warning("No economic indicator data found in data/raw/socioeconomic/")

    if urban_files:
        urban_df = load_urban_density(urban_files[0])
    else:
        logger.warning("No urban density data found in data/raw/socioeconomic/")

    if age_df.empty and econ_df.empty and urban_df.empty:
        logger.warning(
            "No socioeconomic data found. Please download data from ISTAT/Eurostat "
            "and place in data/raw/socioeconomic/"
        )
        return pd.DataFrame()

    merged = merge_socioeconomic_indicators(age_df, econ_df, urban_df)
    save_dataframe(merged, output_path, index=False)
    return merged


if __name__ == "__main__":
    process_socioeconomic_data()
