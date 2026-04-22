"""
Process socioeconomic data for RSVI construction.

Loads and standardizes socioeconomic indicators from Eurostat and OWID at the
NUTS-2 regional level for Italy.  Produces a clean nuts2_code × year panel
covering 2012–2022 with the following columns:

    pct_pop_65plus      – share of regional population aged 65+   (age domain)
    pct_pop_75plus      – share of regional population aged 75+
    pct_pop_80plus      – share of regional population aged 80+
    poverty_rate        – at-risk-of-poverty rate (%)             (economic domain)
    gdp_per_capita      – GDP at current prices, EUR/inhabitant
    disposable_income   – household primary income, EUR/inhabitant
    covid_stringency    – mean June–Sep Oxford Stringency Index for Italy
                          (national proxy; 0 for pre-pandemic years)

Data sources
------------
* eurostat_population_by_age_nuts2.csv  – DEMO_R_PJANGRP3 (2014–2022)
  Missing years 2012-2013 are backward-filled from 2014 (see note below).
  If the file only covers a subset of regions, a warning is logged and the
  age columns are set to NaN for the missing regions.
* eurostat_poverty_rate_nuts2.csv       – ILC_LI41 (2012–2022)
* eurostat_gdp_per_capita_nuts2.csv     – NAMA_10R_2GDP, EUR_HAB (2012–2022)
* eurostat_household_income_nuts2.csv   – NAMA_10R_2HHINC, B6N EUR_HAB (2012–2022)
* owid_covid_global.csv                 – OWID COVID dataset, Italy rows

Backward-fill rationale
-----------------------
DEMO_R_PJANGRP3 starts in 2014.  Age structure (% 65+) is a slow-moving
demographic variable that changes roughly 0.3–0.5 pp per year in Italian
regions.  Assigning 2014 values to 2012–2013 slightly overestimates elderly
share for those years; this is a documented limitation.  2012–2013 had no
major heatwaves, so the impact on regression results is minimal.

Usage
-----
    python -m src.data.process_socioeconomic
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.utils.config import load_config, get_path
from src.utils.constants import NUTS2_CODES, SUMMER_MONTHS
from src.utils.io import save_dataframe

# Full study panel skeleton
_STUDY_YEARS = list(range(2012, 2023))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_eurostat_csv(filepath: Path) -> pd.DataFrame:
    """Read a Eurostat Data Browser CSV, normalising column names.

    Handles two formats produced by the Eurostat Data Browser:
      - Combined: 'geo: Geopolitical entity (reporting)' (code + label in one column)
      - Split:    separate 'geo' (bare code) and 'Geopolitical entity (reporting)' columns

    Returns a DataFrame with standardised columns: geo_code, time_period,
    obs_value, and any extra dimension columns present.
    """
    df = pd.read_csv(filepath)

    rename = {}
    for col in df.columns:
        cl = col.lower().strip()
        # Match geo code column: exactly 'geo' or starts with 'geo:'
        # Avoid matching 'geopolitical entity (reporting)' label columns.
        if cl == "geo" or cl.startswith("geo:"):
            rename[col] = "geo_raw"
        elif cl == "time_period" or cl.startswith("time_period:"):
            rename[col] = "time_period"
        elif cl == "obs_value" or cl.startswith("obs_value:"):
            rename[col] = "obs_value"
    df = df.rename(columns=rename)

    if "geo_raw" in df.columns:
        # Extract bare code from 'ITC1: Piemonte' or bare 'ITC1'
        df["geo_code"] = df["geo_raw"].astype(str).str.split(":").str[0].str.strip()
    if "obs_value" in df.columns:
        df["obs_value"] = pd.to_numeric(df["obs_value"], errors="coerce")

    return df


def _filter_italy(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows for the 20 project NUTS-2 regions."""
    return df[df["geo_code"].isin(NUTS2_CODES)].copy()


def _full_panel_index() -> pd.DataFrame:
    """Return a complete nuts2_code × year skeleton for 2012–2022."""
    rows = [(code, yr) for code in NUTS2_CODES for yr in _STUDY_YEARS]
    return pd.DataFrame(rows, columns=["nuts2_code", "year"])


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_age_structure(filepath: Path) -> pd.DataFrame:
    """Load Eurostat DEMO_R_PJANGRP3 and compute elderly population shares.

    Returns
    -------
    pd.DataFrame
        Columns: nuts2_code, year, pct_pop_65plus, pct_pop_75plus, pct_pop_80plus.
        Covers 2014–2022 from Eurostat; 2012–2013 are backward-filled from 2014.
        Regions absent from the file receive NaN (and a logged warning).
    """
    logger.info(f"Loading age structure from {filepath}")
    df = _parse_eurostat_csv(filepath)

    # Identify the age-class column (named 'age: Age class' or 'age')
    age_col = next((c for c in df.columns if c.lower().startswith("age")), None)
    if age_col is None:
        logger.error("Age class column not found in age structure file")
        return pd.DataFrame()

    # Standardise age codes: strip everything after ':'
    df["age_code"] = df[age_col].astype(str).str.split(":").str[0].str.strip()
    df = _filter_italy(df)
    df["year"] = pd.to_numeric(df["time_period"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["obs_value", "year"])

    # Pivot: rows = (geo_code, year), cols = age codes
    pivot = (
        df.pivot_table(
            index=["geo_code", "year"],
            columns="age_code",
            values="obs_value",
            aggfunc="sum",
        )
        .reset_index()
    )

    # Age groups ≥65: Y65-69, Y70-74, Y75-79, Y80-84, Y_GE85
    ge65_cols = [c for c in pivot.columns if c in {"Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y_GE85"}]
    ge75_cols = [c for c in pivot.columns if c in {"Y75-79", "Y80-84", "Y_GE85"}]
    ge80_cols = [c for c in pivot.columns if c in {"Y80-84", "Y_GE85"}]
    total_col = "TOTAL" if "TOTAL" in pivot.columns else None

    if not total_col:
        logger.error("TOTAL age class not found; cannot compute percentages")
        return pd.DataFrame()

    pivot["pop_ge65"] = pivot[ge65_cols].sum(axis=1)
    pivot["pop_ge75"] = pivot[ge75_cols].sum(axis=1)
    pivot["pop_ge80"] = pivot[ge80_cols].sum(axis=1)
    pivot["pct_pop_65plus"] = pivot["pop_ge65"] / pivot[total_col] * 100
    pivot["pct_pop_75plus"] = pivot["pop_ge75"] / pivot[total_col] * 100
    pivot["pct_pop_80plus"] = pivot["pop_ge80"] / pivot[total_col] * 100

    result = pivot[["geo_code", "year", "pct_pop_65plus", "pct_pop_75plus", "pct_pop_80plus"]].copy()
    result = result.rename(columns={"geo_code": "nuts2_code"})
    result["year"] = result["year"].astype(int)

    # --- Backward-fill 2012–2013 from earliest available year (typically 2014) ---
    available_years = sorted(result["year"].unique())
    earliest = available_years[0]
    missing_years = [y for y in _STUDY_YEARS if y < earliest]
    if missing_years:
        logger.warning(
            f"Age structure data starts at {earliest}; backward-filling "
            f"{missing_years} from {earliest} values. "
            "Documented limitation: see module docstring."
        )
        base = result[result["year"] == earliest].copy()
        frames = [result]
        for yr in missing_years:
            fill = base.copy()
            fill["year"] = yr
            frames.append(fill)
        result = pd.concat(frames, ignore_index=True)

    # Warn about any regions absent from the file
    present = set(result["nuts2_code"].unique())
    missing_regions = set(NUTS2_CODES) - present
    if missing_regions:
        logger.warning(
            f"Age structure file is missing {len(missing_regions)} regions: "
            f"{sorted(missing_regions)}. "
            "Age columns will be NaN for these regions. "
            "Re-download DEMO_R_PJANGRP3 selecting ALL Italian NUTS-2 regions."
        )

    logger.info(
        f"Age structure: {result['nuts2_code'].nunique()} regions, "
        f"years {result['year'].min()}–{result['year'].max()}"
    )
    return result


def load_poverty_rate(filepath: Path) -> pd.DataFrame:
    """Load Eurostat ILC_LI41 at-risk-of-poverty rate by NUTS-2.

    Returns
    -------
    pd.DataFrame
        Columns: nuts2_code, year, poverty_rate.
    """
    logger.info(f"Loading poverty rate from {filepath}")
    df = _parse_eurostat_csv(filepath)
    df = _filter_italy(df)
    df["year"] = pd.to_numeric(df["time_period"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["obs_value", "year"])

    result = (
        df.groupby(["geo_code", "year"], as_index=False)["obs_value"]
        .mean()
        .rename(columns={"geo_code": "nuts2_code", "obs_value": "poverty_rate"})
    )
    result["year"] = result["year"].astype(int)
    logger.info(
        f"Poverty rate: {result['nuts2_code'].nunique()} regions, "
        f"years {result['year'].min()}–{result['year'].max()}"
    )
    return result


def load_gdp_per_capita(filepath: Path) -> pd.DataFrame:
    """Load Eurostat NAMA_10R_2GDP GDP per capita (EUR/inhabitant) by NUTS-2.

    Returns
    -------
    pd.DataFrame
        Columns: nuts2_code, year, gdp_per_capita.
    """
    logger.info(f"Loading GDP per capita from {filepath}")
    df = _parse_eurostat_csv(filepath)

    # Keep EUR_HAB unit if multiple units present
    unit_col = next((c for c in df.columns if "unit" in c.lower()), None)
    if unit_col:
        df = df[df[unit_col].astype(str).str.startswith("EUR_HAB")]

    df = _filter_italy(df)
    df["year"] = pd.to_numeric(df["time_period"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["obs_value", "year"])

    result = (
        df.groupby(["geo_code", "year"], as_index=False)["obs_value"]
        .mean()
        .rename(columns={"geo_code": "nuts2_code", "obs_value": "gdp_per_capita"})
    )
    result["year"] = result["year"].astype(int)
    logger.info(
        f"GDP per capita: {result['nuts2_code'].nunique()} regions, "
        f"years {result['year'].min()}–{result['year'].max()}"
    )
    return result


def load_household_income(filepath: Path) -> pd.DataFrame:
    """Load Eurostat NAMA_10R_2HHINC household disposable income (EUR/inhabitant).

    Uses the B6N (net household disposable income) flow item.

    Returns
    -------
    pd.DataFrame
        Columns: nuts2_code, year, disposable_income.
    """
    logger.info(f"Loading household income from {filepath}")
    df = _parse_eurostat_csv(filepath)

    # Keep disposable income item B6N and EUR_HAB unit
    na_col = next((c for c in df.columns if "na_item" in c.lower()), None)
    unit_col = next((c for c in df.columns if "unit" in c.lower()), None)
    if na_col:
        df = df[df[na_col].astype(str).str.startswith("B6N")]
    if unit_col:
        df = df[df[unit_col].astype(str).str.startswith("EUR_HAB")]

    df = _filter_italy(df)
    df["year"] = pd.to_numeric(df["time_period"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["obs_value", "year"])
    # Restrict to study period
    df = df[df["year"].isin(_STUDY_YEARS)]

    result = (
        df.groupby(["geo_code", "year"], as_index=False)["obs_value"]
        .mean()
        .rename(columns={"geo_code": "nuts2_code", "obs_value": "disposable_income"})
    )
    result["year"] = result["year"].astype(int)
    logger.info(
        f"Household income: {result['nuts2_code'].nunique()} regions, "
        f"years {result['year'].min()}–{result['year'].max()}"
    )
    return result


def load_covid_stringency(filepath: Path) -> pd.DataFrame:
    """Load OWID COVID dataset and compute annual summer stringency for Italy.

    The Oxford Stringency Index (0–100) measures government response intensity.
    Pre-pandemic years receive 0.  Returns a national-level annual series to
    be broadcast to all NUTS-2 regions as a panel control variable.

    Returns
    -------
    pd.DataFrame
        Columns: year, covid_stringency  (national Italy, all 20 NUTS-2 years).
    """
    logger.info(f"Loading COVID stringency from {filepath}")
    df = pd.read_csv(filepath, usecols=["iso_code", "date", "stringency_index"])
    df = df[df["iso_code"] == "ITA"].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    summer = df[df["month"].isin(SUMMER_MONTHS)]
    annual = (
        summer.groupby("year")["stringency_index"]
        .mean()
        .reindex(_STUDY_YEARS, fill_value=0.0)
        .reset_index()
        .rename(columns={"stringency_index": "covid_stringency"})
    )
    # Broadcast to all regions
    panel = _full_panel_index().merge(annual, on="year", how="left")
    panel["covid_stringency"] = panel["covid_stringency"].fillna(0.0)
    logger.info("COVID stringency: Italy national proxy, all years 2012–2022")
    return panel[["nuts2_code", "year", "covid_stringency"]]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_socioeconomic_data(
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Full pipeline: load all socioeconomic files and produce a merged panel.

    Returns
    -------
    pd.DataFrame
        nuts2_code × year panel (220 rows) with all RSVI indicator columns.
    """
    if cfg is None:
        cfg = load_config()

    raw_dir = get_path(cfg, "raw_data") / "socioeconomic"
    output_path = get_path(cfg, "interim_data") / "socioeconomic_processed.parquet"

    # Locate files by descriptive name pattern
    def _first(pattern: str) -> Path | None:
        matches = sorted(raw_dir.glob(pattern))
        return matches[0] if matches else None

    age_file    = _first("*age*")
    poverty_file = _first("*poverty*")
    gdp_file    = _first("*gdp*")
    income_file = _first("*income*")
    covid_file  = _first("*covid*")

    # Start from full panel skeleton so every region-year is represented
    panel = _full_panel_index()

    if age_file:
        age_df = load_age_structure(age_file)
        panel = panel.merge(age_df, on=["nuts2_code", "year"], how="left")
    else:
        logger.warning("No age structure file found (*age*). Age columns will be absent.")

    if poverty_file:
        pov_df = load_poverty_rate(poverty_file)
        panel = panel.merge(pov_df, on=["nuts2_code", "year"], how="left")
    else:
        logger.warning("No poverty rate file found (*poverty*).")

    if gdp_file:
        gdp_df = load_gdp_per_capita(gdp_file)
        panel = panel.merge(gdp_df, on=["nuts2_code", "year"], how="left")
    else:
        logger.warning("No GDP file found (*gdp*).")

    if income_file:
        inc_df = load_household_income(income_file)
        panel = panel.merge(inc_df, on=["nuts2_code", "year"], how="left")
    else:
        logger.warning("No household income file found (*income*).")

    if covid_file:
        covid_df = load_covid_stringency(covid_file)
        panel = panel.merge(covid_df, on=["nuts2_code", "year"], how="left")
    else:
        logger.warning("No COVID file found (*covid*). covid_stringency will be absent.")

    panel = panel.sort_values(["nuts2_code", "year"]).reset_index(drop=True)

    # Summary
    logger.info(f"Socioeconomic panel: {len(panel)} rows, {len(panel.columns)} columns")
    null_summary = panel.isnull().sum()
    if null_summary.any():
        logger.info(f"Null counts:\n{null_summary[null_summary > 0].to_string()}")

    save_dataframe(panel, output_path, index=False)
    return panel


if __name__ == "__main__":
    process_socioeconomic_data()
