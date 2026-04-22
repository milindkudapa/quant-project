"""
Weekly climate aggregation with non-linear heat indicators.

Aggregates daily regional climate and heatwave flags to an ISO-week panel
and computes the features needed for the non-linear weekly regression:
- tmax_max_week, tmax_mean_week: weekly temperature summaries
- tmax_anomaly_week: deviation from (region, week-of-year) climatology
- tmax_anomaly_week_sq: quadratic term (captures non-linear heat response)
- above_p90_days, above_p95_days: threshold exceedance counts per week
- hw_days_week: heatwave days within the ISO week

Usage
-----
    python -m src.features.weekly_climate
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import load_config, get_path
from src.utils.io import load_dataframe, save_dataframe


def _attach_iso_week(df: pd.DataFrame) -> pd.DataFrame:
    iso = df["date"].dt.isocalendar()
    out = df.copy()
    out["iso_year"] = iso["year"].astype(int)
    out["iso_week"] = iso["week"].astype(int)
    return out


def compute_region_week_climatology(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean Tmax per (region, week-of-year) across all study years.

    Returns a DataFrame indexed by (nuts2_code, iso_week) with column
    `tmax_climatology`.
    """
    weekly_mean = (
        daily.groupby(["nuts2_code", "iso_year", "iso_week"])["tmax"]
        .mean()
        .reset_index()
    )
    clim = (
        weekly_mean.groupby(["nuts2_code", "iso_week"])["tmax"]
        .mean()
        .rename("tmax_climatology")
        .reset_index()
    )
    logger.info(
        f"Climatology computed for {clim['nuts2_code'].nunique()} regions × "
        f"{clim['iso_week'].nunique()} weeks-of-year"
    )
    return clim


def compute_region_percentiles(daily: pd.DataFrame) -> pd.DataFrame:
    """Region-specific 90th and 95th percentile of daily Tmax (summer baseline)."""
    pcts = (
        daily.groupby("nuts2_code")["tmax"]
        .quantile([0.90, 0.95])
        .unstack()
        .rename(columns={0.90: "tmax_p90", 0.95: "tmax_p95"})
        .reset_index()
    )
    return pcts


def build_weekly_climate(
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the weekly climate panel.

    Returns
    -------
    pd.DataFrame
        One row per (nuts2_code, iso_year, iso_week) with heat features.
    """
    if cfg is None:
        cfg = load_config()

    interim = get_path(cfg, "interim_data")
    daily = load_dataframe(interim / "daily_regional_climate.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = _attach_iso_week(daily)

    # Heatwave flags already computed in build_heatwave_features
    hw_flags_path = interim / "daily_heatwave_flags.parquet"
    if hw_flags_path.exists():
        hw = load_dataframe(hw_flags_path)
        hw["date"] = pd.to_datetime(hw["date"])
        hw = _attach_iso_week(hw)[
            ["nuts2_code", "iso_year", "iso_week", "date", "is_heatwave_day"]
        ]
        daily = daily.merge(
            hw[["nuts2_code", "date", "is_heatwave_day"]],
            on=["nuts2_code", "date"],
            how="left",
        )
        daily["is_heatwave_day"] = daily["is_heatwave_day"].fillna(False)
    else:
        logger.warning("Daily heatwave flags not found; hw_days_week will be 0")
        daily["is_heatwave_day"] = False

    # Region-level percentile thresholds (for weekly exceedance counts)
    pcts = compute_region_percentiles(daily)
    daily = daily.merge(pcts, on="nuts2_code", how="left")
    daily["above_p90"] = (daily["tmax"] > daily["tmax_p90"]).astype(int)
    daily["above_p95"] = (daily["tmax"] > daily["tmax_p95"]).astype(int)

    # Aggregate to weekly
    grp = daily.groupby(["nuts2_code", "iso_year", "iso_week"])
    weekly = grp.agg(
        tmax_max_week=("tmax", "max"),
        tmax_mean_week=("tmax", "mean"),
        tmin_mean_week=("tmin", "mean"),
        tmean_week=("tmean", "mean"),
        days_in_week=("tmax", "size"),
        hw_days_week=("is_heatwave_day", "sum"),
        above_p90_days=("above_p90", "sum"),
        above_p95_days=("above_p95", "sum"),
    ).reset_index()

    # Drop weeks with fewer than 4 observed days (partial weeks at boundaries)
    weekly = weekly[weekly["days_in_week"] >= 4].copy()

    # Climatology and anomaly
    clim = compute_region_week_climatology(daily)
    weekly = weekly.merge(clim, on=["nuts2_code", "iso_week"], how="left")
    weekly["tmax_anomaly_week"] = weekly["tmax_mean_week"] - weekly["tmax_climatology"]
    weekly["tmax_anomaly_week_sq"] = weekly["tmax_anomaly_week"] ** 2

    # Useful flags
    weekly["any_hw_day"] = (weekly["hw_days_week"] > 0).astype(int)

    weekly = weekly.sort_values(["nuts2_code", "iso_year", "iso_week"]).reset_index(
        drop=True
    )
    out_path = interim / "weekly_regional_climate.parquet"
    save_dataframe(weekly, out_path, index=False)
    logger.success(
        f"Weekly climate panel: {len(weekly)} rows, "
        f"{weekly['nuts2_code'].nunique()} regions, "
        f"{weekly['iso_year'].nunique()} years, "
        f"weeks {weekly['iso_week'].min()}–{weekly['iso_week'].max()}"
    )
    return weekly


if __name__ == "__main__":
    build_weekly_climate()
