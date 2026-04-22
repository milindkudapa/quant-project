"""
Process mortality data from ISTAT / Eurostat sources.

Handles loading, cleaning, and aggregating mortality data to NUTS-2 regional
level for use in the panel analysis.

Usage
-----
    python -m src.data.process_mortality
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import load_config, get_path, get_region_mapping
from src.utils.constants import SUMMER_MONTHS, NUTS2_CODES
from src.utils.io import save_dataframe


def load_istat_mortality(filepath: Path) -> pd.DataFrame:
    """Load ISTAT daily mortality data.

    Expects a CSV with columns for date, geographic area, age group, sex,
    and death counts. The exact format depends on the ISTAT download.

    Parameters
    ----------
    filepath : Path
        Path to the ISTAT mortality CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned mortality data with standardized column names.
    """
    logger.info(f"Loading ISTAT mortality data from {filepath}")

    df = pd.read_csv(filepath)

    # Standardize column names (adjust based on actual ISTAT file format)
    rename_map = {
        "DATA": "date",
        "TERRITORIO": "territory",
        "CLASSE_ETA": "age_group",
        "SESSO": "sex",
        "DECESSI": "deaths",
    }
    # Try common ISTAT column name patterns
    for old, new in rename_map.items():
        matching = [c for c in df.columns if old.lower() in c.lower()]
        if matching:
            df = df.rename(columns={matching[0]: new})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    logger.info(f"Loaded {len(df)} mortality records")
    return df


def load_eurostat_mortality(filepath: Path) -> pd.DataFrame:
    """Load Eurostat weekly mortality data for Italian NUTS-2 regions.

    Parameters
    ----------
    filepath : Path
        Path to the Eurostat mortality file (TSV or CSV).

    Returns
    -------
    pd.DataFrame
        Cleaned weekly mortality data.
    """
    logger.info(f"Loading Eurostat mortality data from {filepath}")

    # Eurostat files can be TSV with complex headers
    if filepath.suffix == ".tsv":
        df = pd.read_csv(filepath, sep="\t")
    else:
        df = pd.read_csv(filepath)

    # Standardize Eurostat Data Browser format if present
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if "geo:" in col_lower or "geo" == col_lower:
            rename_map[col] = "nuts2_code"
        elif "time_period" in col_lower:
            rename_map[col] = "time_period"
        elif "obs_value" in col_lower:
            rename_map[col] = "deaths"
        elif "sex" in col_lower:
            rename_map[col] = "sex"
            
    df = df.rename(columns=rename_map)

    if "nuts2_code" in df.columns:
        # Extract just the code from 'ITG1: Sicilia'
        df["nuts2_code"] = df["nuts2_code"].astype(str).str.split(":").str[0].str.strip()
        # Merge ITH2 (Trento) into ITH1 (Trentino-Alto Adige) to match ISTAT's
        # single-region reporting and the merged ITH1 boundary in climate data.
        df.loc[df["nuts2_code"] == "ITH2", "nuts2_code"] = "ITH1"
        # Filter to only Italian NUTS-2 regions (now includes ITH5=Emilia-Romagna)
        df = df[df["nuts2_code"].isin(NUTS2_CODES)].copy()

    if "time_period" in df.columns:
        # Convert weekly format '2015-W01' to a datetime (Monday of that week)
        is_weekly = df["time_period"].astype(str).str.contains("-W").any()
        if is_weekly:
            time_str = df["time_period"].astype(str) + "-1"
            df["date"] = pd.to_datetime(time_str, format="%G-W%V-%u", errors="coerce")
        else:
            df["date"] = pd.to_datetime(df["time_period"], errors="coerce")

    # Clean deaths column (sometimes has characters or spaces in raw Eurostat)
    if "deaths" in df.columns:
        df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce")
        df = df.dropna(subset=["deaths"])

    logger.info(f"Loaded {len(df)} Eurostat mortality records (Italian NUTS-2 filtered)")
    return df


def aggregate_to_nuts2_monthly(
    df: pd.DataFrame,
    region_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate mortality data to NUTS-2 monthly level.

    Parameters
    ----------
    df : pd.DataFrame
        Raw mortality data with date, territory, and deaths columns.
    region_mapping : dict, optional
        Municipality-to-NUTS2 mapping lookup.

    Returns
    -------
    pd.DataFrame
        Monthly mortality counts per NUTS-2 region, age group, and sex.
    """
    df = df.copy()

    # Extract year and month
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Aggregate — the exact groupby depends on available geographic resolution
    # If data is already at regional level:
    group_cols = ["nuts2_code", "year", "month"]
    if "age_group" in df.columns:
        group_cols.append("age_group")
    if "sex" in df.columns:
        group_cols.append("sex")

    monthly = (
        df.groupby(group_cols, as_index=False)["deaths"]
        .sum()
    )

    logger.info(f"Aggregated to {len(monthly)} region-month records")
    return monthly


def compute_summer_mortality(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compute summer (June–September) total mortality per region-year.

    Parameters
    ----------
    monthly : pd.DataFrame
        Monthly mortality with columns: nuts2_code, year, month, deaths.

    Returns
    -------
    pd.DataFrame
        Summer mortality per region-year.
    """
    summer = monthly[monthly["month"].isin(SUMMER_MONTHS)].copy()

    # Aggregate across summer months (and optionally across age/sex for totals)
    group_cols = ["nuts2_code", "year"]
    summer_totals = (
        summer.groupby(group_cols, as_index=False)["deaths"]
        .sum()
        .rename(columns={"deaths": "summer_deaths"})
    )

    logger.info(f"Computed summer mortality for {len(summer_totals)} region-years")
    return summer_totals


def process_mortality_data(
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Full pipeline: load and process mortality data.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Processed mortality data ready for panel assembly.
    """
    if cfg is None:
        cfg = load_config()

    raw_dir = get_path(cfg, "raw_data") / "mortality"
    output_path = get_path(cfg, "interim_data") / "mortality_processed.parquet"

    # Try ISTAT data first, fall back to Eurostat
    raw_files = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.xlsx"))

    if not raw_files:
        logger.warning("No mortality data files found in data/raw/mortality/")
        logger.info(
            "Please download mortality data from ISTAT or Eurostat and "
            "place it in data/raw/mortality/"
        )
        return pd.DataFrame()

    # Load and process each file
    all_frames = []
    for f in raw_files:
        try:
            # We guess the format based on columns inside the loader
            if "eurostat" in f.name.lower() or "demo_r" in f.name.lower() or "ts" in f.name.lower():
                df = load_eurostat_mortality(f)
            else:
                df = load_istat_mortality(f) if f.suffix == ".csv" else pd.read_excel(f)
                
            all_frames.append(df)
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)

    # Aggregate to monthly and compute summer totals
    if "date" in combined.columns and "nuts2_code" in combined.columns:
        monthly = aggregate_to_nuts2_monthly(combined)
        summer = compute_summer_mortality(monthly)
        save_dataframe(summer, output_path, index=False)
        save_dataframe(
            summer, 
            get_path(cfg, "interim_data") / "mortality_processed_eurostat.parquet", 
            index=False
        )
        return summer

    # If columns don't match expected format, save as-is for manual inspection
    interim_path = get_path(cfg, "interim_data") / "mortality_raw_combined.parquet"
    save_dataframe(combined, interim_path, index=False)
    logger.warning(
        "Mortality data columns don't match expected format. "
        f"Saved raw combined data to {interim_path} for inspection."
    )
    return combined


if __name__ == "__main__":
    process_mortality_data()
