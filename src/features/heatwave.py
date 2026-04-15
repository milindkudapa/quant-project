"""
Heatwave detection and metrics.

Identifies heatwave events based on region-specific temperature thresholds
(90th percentile of daily Tmax over the reference period, ≥3 consecutive days)
and computes heatwave summary statistics for each region-summer.

Usage
-----
    python -m src.features.heatwave
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import load_config, get_path
from src.utils.constants import SUMMER_MONTHS
from src.utils.io import load_dataframe, save_dataframe


def compute_percentile_thresholds(
    daily_climate: pd.DataFrame,
    percentile: int = 90,
    reference_start: int = 2012,
    reference_end: int = 2022,
) -> pd.Series:
    """Compute region-specific temperature percentile thresholds.

    Calculates the Nth percentile of daily Tmax during summer months
    (June–September) over the reference period for each region.

    Parameters
    ----------
    daily_climate : pd.DataFrame
        Daily climate data with columns: date, nuts2_code, tmax.
    percentile : int
        Percentile threshold (default: 90).
    reference_start : int
        Start year of the reference period.
    reference_end : int
        End year of the reference period.

    Returns
    -------
    pd.Series
        Percentile threshold indexed by nuts2_code.
    """
    df = daily_climate.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Filter to reference period and summer months
    mask = (
        (df["year"] >= reference_start)
        & (df["year"] <= reference_end)
        & (df["month"].isin(SUMMER_MONTHS))
    )
    ref = df[mask]

    thresholds = ref.groupby("nuts2_code")["tmax"].quantile(percentile / 100)
    thresholds.name = f"tmax_p{percentile}"

    logger.info(
        f"Computed {percentile}th percentile thresholds for "
        f"{len(thresholds)} regions "
        f"(range: {thresholds.min():.1f}°C – {thresholds.max():.1f}°C)"
    )
    return thresholds


def detect_heatwave_days(
    daily_climate: pd.DataFrame,
    thresholds: pd.Series,
    min_consecutive: int = 3,
) -> pd.DataFrame:
    """Identify heatwave days based on consecutive exceedance.

    A day is classified as a heatwave day if:
    - The daily Tmax exceeds the region-specific percentile threshold, AND
    - It is part of a run of ≥ `min_consecutive` consecutive exceedance days.

    Parameters
    ----------
    daily_climate : pd.DataFrame
        Daily climate data with columns: date, nuts2_code, tmax.
    thresholds : pd.Series
        Region-specific temperature thresholds (indexed by nuts2_code).
    min_consecutive : int
        Minimum number of consecutive days above threshold (default: 3).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional columns:
        - ``above_threshold``: bool, whether Tmax exceeds the threshold
        - ``is_heatwave_day``: bool, whether the day is part of a heatwave
        - ``hw_event_id``: int, unique heatwave event identifier per region
    """
    df = daily_climate.copy()
    df = df.sort_values(["nuts2_code", "date"]).reset_index(drop=True)

    # Merge thresholds
    df["threshold"] = df["nuts2_code"].map(thresholds)
    df["above_threshold"] = df["tmax"] > df["threshold"]

    # Detect consecutive runs within each region
    df["is_heatwave_day"] = False
    df["hw_event_id"] = 0

    event_counter = 0

    for region in df["nuts2_code"].unique():
        mask = df["nuts2_code"] == region
        region_df = df.loc[mask].copy()

        # Find consecutive runs of above-threshold days
        above = region_df["above_threshold"].values
        run_lengths = []
        run_starts = []

        i = 0
        while i < len(above):
            if above[i]:
                start = i
                while i < len(above) and above[i]:
                    i += 1
                run_length = i - start
                run_lengths.append(run_length)
                run_starts.append(start)
            else:
                i += 1

        # Mark heatwave days (runs ≥ min_consecutive)
        hw_mask = np.zeros(len(region_df), dtype=bool)
        hw_ids = np.zeros(len(region_df), dtype=int)

        for start, length in zip(run_starts, run_lengths):
            if length >= min_consecutive:
                event_counter += 1
                hw_mask[start : start + length] = True
                hw_ids[start : start + length] = event_counter

        df.loc[mask, "is_heatwave_day"] = hw_mask
        df.loc[mask, "hw_event_id"] = hw_ids

    n_hw_days = df["is_heatwave_day"].sum()
    n_events = df["hw_event_id"].max()
    logger.info(
        f"Detected {n_hw_days} heatwave days across {n_events} events "
        f"(threshold: {min_consecutive}+ consecutive days)"
    )

    return df


def compute_heatwave_metrics(
    hw_daily: pd.DataFrame,
) -> pd.DataFrame:
    """Compute seasonal heatwave summary metrics per region-year.

    Parameters
    ----------
    hw_daily : pd.DataFrame
        Daily data with heatwave flags from :func:`detect_heatwave_days`.

    Returns
    -------
    pd.DataFrame
        Region-year summary with columns:
        - ``hw_days``: total heatwave days in summer
        - ``hw_events``: number of distinct heatwave events
        - ``hw_max_duration``: longest heatwave event (days)
        - ``hw_intensity``: mean Tmax anomaly during heatwave days (°C above threshold)
        - ``summer_tmax_mean``: mean daily Tmax over summer
        - ``summer_tmax_anomaly``: deviation from the region's long-term summer Tmax mean
    """
    df = hw_daily.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Filter to summer months
    summer = df[df["month"].isin(SUMMER_MONTHS)].copy()

    # Compute metrics per region-year
    metrics = []

    for (region, year), group in summer.groupby(["nuts2_code", "year"]):
        hw_subset = group[group["is_heatwave_day"]]

        rec = {
            "nuts2_code": region,
            "year": year,
            "hw_days": len(hw_subset),
            "hw_events": hw_subset["hw_event_id"].nunique() if len(hw_subset) > 0 else 0,
            "hw_max_duration": 0,
            "hw_intensity": 0.0,
            "summer_tmax_mean": group["tmax"].mean(),
        }

        # Max duration of a single heatwave event
        if len(hw_subset) > 0:
            event_durations = hw_subset.groupby("hw_event_id").size()
            rec["hw_max_duration"] = event_durations.max()
            rec["hw_intensity"] = (hw_subset["tmax"] - hw_subset["threshold"]).mean()

        metrics.append(rec)

    metrics_df = pd.DataFrame(metrics)

    # Compute summer Tmax anomaly (deviation from region's long-term mean)
    region_means = metrics_df.groupby("nuts2_code")["summer_tmax_mean"].mean()
    metrics_df["summer_tmax_anomaly"] = metrics_df.apply(
        lambda row: row["summer_tmax_mean"] - region_means[row["nuts2_code"]],
        axis=1,
    )

    logger.info(f"Computed heatwave metrics for {len(metrics_df)} region-years")
    return metrics_df


def build_heatwave_features(
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Full pipeline: load climate data → compute heatwave features.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Heatwave metrics per region-year.
    """
    if cfg is None:
        cfg = load_config()

    climate_path = get_path(cfg, "interim_data") / "daily_regional_climate.parquet"
    output_path = get_path(cfg, "interim_data") / "heatwave_metrics.parquet"

    daily = load_dataframe(climate_path)
    daily["date"] = pd.to_datetime(daily["date"])

    hw_cfg = cfg["heatwave"]

    # Compute thresholds
    thresholds = compute_percentile_thresholds(
        daily,
        percentile=hw_cfg["percentile_threshold"],
        reference_start=hw_cfg["reference_start_year"],
        reference_end=hw_cfg["reference_end_year"],
    )

    # Detect heatwave days
    hw_daily = detect_heatwave_days(
        daily,
        thresholds,
        min_consecutive=hw_cfg["min_consecutive_days"],
    )

    # Compute seasonal metrics
    metrics = compute_heatwave_metrics(hw_daily)

    save_dataframe(metrics, output_path, index=False)

    # Also save daily heatwave flags for potential sub-analyses
    hw_daily_path = get_path(cfg, "interim_data") / "daily_heatwave_flags.parquet"
    save_dataframe(hw_daily, hw_daily_path, index=False)

    return metrics


if __name__ == "__main__":
    build_heatwave_features()
