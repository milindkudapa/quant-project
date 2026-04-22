"""
Weekly panel assembly.

Merges weekly climate + weekly mortality + (annual) RSVI into a
region-week panel. Adds heat-exposure lags, non-linear terms, and
interactions needed for the weekly H1/H2/H3 regressions.

Usage
-----
    python -m src.analysis.weekly_panel_dataset
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import load_config, get_path
from src.utils.io import load_dataframe, save_dataframe


LAG_VARS = ["hw_days_week", "tmax_anomaly_week", "above_p90_days", "above_p95_days"]
LAGS = (1, 2)


def add_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """Add 1-week and 2-week lags of the heat-exposure variables.

    Lags are computed within region, ordered by (iso_year, iso_week).
    Lags that cross into the previous summer (> 2 weeks apart) are set to
    NaN to avoid leakage from W39 of year Y-1 into W22 of year Y.
    """
    df = panel.sort_values(["nuts2_code", "iso_year", "iso_week"]).copy()
    df["_week_index"] = df["iso_year"] * 100 + df["iso_week"]
    for var in LAG_VARS:
        for lag in LAGS:
            lag_col = f"{var}_lag{lag}"
            df[lag_col] = df.groupby("nuts2_code")[var].shift(lag)
            prev_idx = df.groupby("nuts2_code")["_week_index"].shift(lag)
            # If gap between current week and lagged week > lag weeks, drop it.
            # Weeks within a summer increment by 1; across summers, gap is ~63
            # (e.g. 201239 → 201322 → _week_index difference = 83). So any gap
            # > lag * 2 indicates a summer boundary crossing.
            gap_ok = (df["_week_index"] - prev_idx) <= lag * 2
            df.loc[~gap_ok.fillna(False), lag_col] = np.nan
    df = df.drop(columns=["_week_index"])
    return df


def add_nonlinear_and_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["d2022"] = (df["iso_year"] == 2022).astype(int)
    df["covid_period"] = df["iso_year"].isin([2020, 2021, 2022]).astype(int)
    df["covid_2020_2021"] = df["iso_year"].isin([2020, 2021]).astype(int)

    # Main non-linear heat terms already created in weekly_climate
    # Threshold indicators (1 if any exceedance day in the week)
    df["hot_week_p90"] = (df["above_p90_days"] > 0).astype(int)
    df["hot_week_p95"] = (df["above_p95_days"] > 0).astype(int)

    # Interaction terms with RSVI
    df["hw_days_x_rsvi"] = df["hw_days_week"] * df["rsvi"]
    df["tmax_anom_x_rsvi"] = df["tmax_anomaly_week"] * df["rsvi"]
    df["tmax_anom_sq_x_rsvi"] = df["tmax_anomaly_week_sq"] * df["rsvi"]
    df["p95_x_rsvi"] = df["hot_week_p95"] * df["rsvi"]

    # Triple interactions with 2022 dummy
    df["hw_days_x_rsvi_x_d2022"] = df["hw_days_x_rsvi"] * df["d2022"]
    df["tmax_anom_x_rsvi_x_d2022"] = df["tmax_anom_x_rsvi"] * df["d2022"]
    df["p95_x_rsvi_x_d2022"] = df["p95_x_rsvi"] * df["d2022"]

    # Lagged interactions (same-week + distributed lag × rsvi)
    for lag in LAGS:
        df[f"hw_days_week_lag{lag}_x_rsvi"] = (
            df[f"hw_days_week_lag{lag}"] * df["rsvi"]
        )
    return df


def build_weekly_panel(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()

    interim = get_path(cfg, "interim_data")
    processed = get_path(cfg, "processed_data")

    climate = load_dataframe(interim / "weekly_regional_climate.parquet")
    mortality = load_dataframe(interim / "weekly_mortality.parquet")
    rsvi = load_dataframe(interim / "rsvi.parquet")[["nuts2_code", "year", "rsvi"]]

    panel = climate.merge(
        mortality[
            [
                "nuts2_code",
                "iso_year",
                "iso_week",
                "deaths",
                "population",
                "mortality_rate_week",
            ]
        ],
        on=["nuts2_code", "iso_year", "iso_week"],
        how="inner",
    )

    # Annual RSVI propagated to all weeks within the year
    panel = panel.merge(
        rsvi, left_on=["nuts2_code", "iso_year"], right_on=["nuts2_code", "year"],
        how="left",
    ).drop(columns=["year"])

    panel = add_lags(panel)
    panel = add_nonlinear_and_interactions(panel)

    # Log-transform outcome
    panel["log_mortality_rate_week"] = np.log(panel["mortality_rate_week"])

    # A compact integer time index for linearmodels MultiIndex uniqueness
    panel["week_id"] = panel["iso_year"] * 100 + panel["iso_week"]

    panel = panel.sort_values(["nuts2_code", "week_id"]).reset_index(drop=True)

    out_csv = processed / "weekly_panel_dataset.csv"
    out_pq = processed / "weekly_panel_dataset.parquet"
    save_dataframe(panel, out_csv, index=False)
    save_dataframe(panel, out_pq, index=False)

    logger.success(
        f"Weekly panel: {len(panel)} rows, "
        f"{panel['nuts2_code'].nunique()} regions × "
        f"{panel['iso_year'].nunique()} years × "
        f"{panel['iso_week'].nunique()} weeks-of-year. "
        f"Columns: {len(panel.columns)}"
    )
    return panel


if __name__ == "__main__":
    build_weekly_panel()
