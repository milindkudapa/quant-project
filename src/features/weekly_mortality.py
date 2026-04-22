"""
Weekly mortality from Eurostat NUTS-2 all-cause deaths.

Parses the Eurostat weekly deaths file, filters to Italian NUTS-2,
merges ITH2 into ITH1 (to match the annual pipeline and climate panel),
attaches annual population for rate computation, and keeps summer weeks
(W22–W39) to align with the climate panel.

Usage
-----
    python -m src.features.weekly_mortality
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

from src.utils.config import load_config, get_path
from src.utils.constants import NUTS2_CODES
from src.utils.io import load_dataframe, save_dataframe


def _parse_iso_week(time_str: pd.Series) -> pd.DataFrame:
    parts = time_str.str.split("-W", expand=True)
    return pd.DataFrame(
        {
            "iso_year": parts[0].astype(int),
            "iso_week": parts[1].astype(int),
        }
    )


def load_weekly_mortality_raw(cfg: dict[str, Any]) -> pd.DataFrame:
    path = (
        get_path(cfg, "raw_data")
        / "mortality"
        / "eurostat_weekly_mortality_nuts2.csv"
    )
    raw = load_dataframe(path)

    it = raw[
        raw["geo: Geopolitical entity (reporting)"].str.startswith("IT", na=False)
        & (raw["sex: Sex"] == "T: Total")
    ].copy()

    it["nuts2_code"] = (
        it["geo: Geopolitical entity (reporting)"].str.split(":").str[0].str.strip()
    )
    # Merge Trento (ITH2) into Bolzano+Trento (ITH1) to mirror the annual pipeline
    it.loc[it["nuts2_code"] == "ITH2", "nuts2_code"] = "ITH1"

    it = it[it["nuts2_code"].isin(NUTS2_CODES)].copy()

    iso = _parse_iso_week(it["TIME_PERIOD: Time"])
    it["iso_year"] = iso["iso_year"].values
    it["iso_week"] = iso["iso_week"].values
    it["deaths"] = pd.to_numeric(it["OBS_VALUE: Observation value"], errors="coerce")

    out = (
        it.groupby(["nuts2_code", "iso_year", "iso_week"], as_index=False)["deaths"]
        .sum(min_count=1)
    )
    logger.info(
        f"Parsed weekly mortality: {len(out)} rows, "
        f"{out['nuts2_code'].nunique()} regions, "
        f"years {out['iso_year'].min()}–{out['iso_year'].max()}"
    )
    return out


def load_annual_population(cfg: dict[str, Any]) -> pd.DataFrame:
    # Re-use the annual panel's population (already saved after backfill)
    panel_path = get_path(cfg, "processed_data") / "panel_dataset.parquet"
    panel = load_dataframe(panel_path)
    pop = panel[["nuts2_code", "year", "population"]].drop_duplicates()
    return pop


def build_weekly_mortality(
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()

    study_start = cfg["study"]["start_year"]
    study_end = cfg["study"]["end_year"]

    weekly = load_weekly_mortality_raw(cfg)
    weekly = weekly[
        weekly["iso_year"].between(study_start, study_end)
        & weekly["iso_week"].between(22, 39)
    ].copy()

    pop = load_annual_population(cfg)
    weekly = weekly.merge(
        pop, left_on=["nuts2_code", "iso_year"], right_on=["nuts2_code", "year"],
        how="left",
    ).drop(columns=["year"])

    # Weekly mortality rate per 100k: weekly_deaths / (population/52) * 100k
    weekly["mortality_rate_week"] = (
        weekly["deaths"] / (weekly["population"] / 52.0) * 100_000
    )

    out_path = get_path(cfg, "interim_data") / "weekly_mortality.parquet"
    save_dataframe(weekly, out_path, index=False)
    logger.success(
        f"Weekly mortality: {len(weekly)} rows, "
        f"{weekly['nuts2_code'].nunique()} regions, "
        f"{weekly['iso_year'].nunique()} years"
    )
    return weekly


if __name__ == "__main__":
    build_weekly_mortality()
