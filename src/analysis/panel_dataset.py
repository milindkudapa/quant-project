"""
Panel dataset assembly.

Merges all processed data sources (mortality, heatwave metrics, RSVI,
population) into the final panel dataset for regression analysis.

Usage
-----
    python -m src.analysis.panel_dataset
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from src.utils.config import load_config, get_path, get_region_mapping
from src.utils.io import load_dataframe, save_dataframe


def merge_panel_components(
    mortality: pd.DataFrame,
    heatwave: pd.DataFrame,
    rsvi: pd.DataFrame,
    population: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge all data components into a single panel dataset.

    Parameters
    ----------
    mortality : pd.DataFrame
        Summer mortality data with columns: nuts2_code, year, summer_deaths.
    heatwave : pd.DataFrame
        Heatwave metrics with columns: nuts2_code, year, hw_days, etc.
    rsvi : pd.DataFrame
        RSVI data with columns: nuts2_code, year, rsvi.
    population : pd.DataFrame, optional
        Population data with columns: nuts2_code, year, population.

    Returns
    -------
    pd.DataFrame
        Merged panel dataset.
    """
    merge_keys = ["nuts2_code", "year"]

    # Start with heatwave data (most complete temporal coverage)
    panel = heatwave.copy()

    # Merge mortality
    if not mortality.empty:
        panel = panel.merge(mortality, on=merge_keys, how="left")
        logger.info(f"Merged mortality data: {mortality.shape}")

    # Merge RSVI
    if not rsvi.empty:
        panel = panel.merge(rsvi, on=merge_keys, how="left")
        logger.info(f"Merged RSVI data: {rsvi.shape}")

    # Merge population
    if population is not None and not population.empty:
        panel = panel.merge(population, on=merge_keys, how="left")
        logger.info(f"Merged population data: {population.shape}")

    logger.info(
        f"Panel dataset: {len(panel)} observations, "
        f"{panel['nuts2_code'].nunique()} regions, "
        f"{panel['year'].nunique()} years"
    )
    return panel


def add_derived_variables(panel: pd.DataFrame) -> pd.DataFrame:
    """Add derived variables needed for the regression models.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset with base variables.

    Returns
    -------
    pd.DataFrame
        Panel with added derived variables.
    """
    df = panel.copy()

    # Mortality rate (per 100,000 population)
    if "summer_deaths" in df.columns and "population" in df.columns:
        df["mortality_rate"] = (df["summer_deaths"] / df["population"]) * 100_000

    # 2022 dummy variable
    df["d2022"] = (df["year"] == 2022).astype(int)

    # COVID period indicator
    df["covid_period"] = df["year"].isin([2020, 2021, 2022]).astype(int)

    # Interaction terms
    if "hw_days" in df.columns and "rsvi" in df.columns:
        df["hw_days_x_rsvi"] = df["hw_days"] * df["rsvi"]

    if "summer_tmax_anomaly" in df.columns and "rsvi" in df.columns:
        df["tmax_anomaly_x_rsvi"] = df["summer_tmax_anomaly"] * df["rsvi"]

    # Three-way interaction for H3
    if "hw_days_x_rsvi" in df.columns:
        df["hw_days_x_rsvi_x_d2022"] = df["hw_days_x_rsvi"] * df["d2022"]

    if "tmax_anomaly_x_rsvi" in df.columns:
        df["tmax_anomaly_x_rsvi_x_d2022"] = df["tmax_anomaly_x_rsvi"] * df["d2022"]

    # Region name mapping
    from src.utils.config import load_config

    cfg = load_config()
    region_names = get_region_mapping(cfg)
    df["region_name"] = df["nuts2_code"].map(region_names)

    logger.info(f"Added derived variables. Panel columns: {df.columns.tolist()}")
    return df


def build_panel_dataset(
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Full pipeline: load all interim data → assemble panel dataset.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Complete panel dataset ready for regression analysis.
    """
    if cfg is None:
        cfg = load_config()

    interim_dir = get_path(cfg, "interim_data")
    output_path = get_path(cfg, "processed_data") / "panel_dataset.csv"

    # Load components
    heatwave = pd.DataFrame()
    mortality = pd.DataFrame()
    rsvi = pd.DataFrame()

    hw_path = interim_dir / "heatwave_metrics.parquet"
    if hw_path.exists():
        heatwave = load_dataframe(hw_path)
    else:
        logger.warning(f"Heatwave metrics not found: {hw_path}")

    mort_path = interim_dir / "mortality_processed.parquet"
    if mort_path.exists():
        mortality = load_dataframe(mort_path)
    else:
        logger.warning(f"Mortality data not found: {mort_path}")

    rsvi_path = interim_dir / "rsvi.parquet"
    if rsvi_path.exists():
        rsvi = load_dataframe(rsvi_path)
    else:
        logger.warning(f"RSVI data not found: {rsvi_path}")

    if heatwave.empty:
        raise FileNotFoundError(
            "Heatwave metrics are required. Run feature engineering first."
        )

    # Merge
    panel = merge_panel_components(mortality, heatwave, rsvi)
    panel = add_derived_variables(panel)

    # Sort and save
    panel = panel.sort_values(["nuts2_code", "year"]).reset_index(drop=True)
    save_dataframe(panel, output_path, index=False)

    # Also save as parquet for faster loading
    save_dataframe(panel, output_path.with_suffix(".parquet"), index=False)

    return panel


if __name__ == "__main__":
    build_panel_dataset()
