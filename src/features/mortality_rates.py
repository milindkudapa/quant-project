"""
Mortality rate computation.

Computes age-standardized mortality rates using the European Standard
Population, and excess mortality relative to baseline expectations.

Usage
-----
    python -m src.features.mortality_rates
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.constants import EUROPEAN_STANDARD_POPULATION


def compute_crude_mortality_rate(
    deaths: pd.Series,
    population: pd.Series,
    per: int = 100_000,
) -> pd.Series:
    """Compute crude mortality rate.

    Parameters
    ----------
    deaths : pd.Series
        Number of deaths.
    population : pd.Series
        Total population.
    per : int
        Rate denominator (default: per 100,000).

    Returns
    -------
    pd.Series
        Crude mortality rate.
    """
    return (deaths / population) * per


def compute_age_standardized_rate(
    deaths_by_age: pd.DataFrame,
    population_by_age: pd.DataFrame,
    standard_pop: dict[str, int] | None = None,
    per: int = 100_000,
) -> pd.Series:
    """Compute age-standardized mortality rate (direct method).

    Uses the European Standard Population (2013) as the reference.

    Parameters
    ----------
    deaths_by_age : pd.DataFrame
        Deaths with columns for each age group.
    population_by_age : pd.DataFrame
        Population with columns for each age group.
    standard_pop : dict, optional
        Standard population weights by age group. Defaults to the
        European Standard Population.
    per : int
        Rate denominator (default: per 100,000).

    Returns
    -------
    pd.Series
        Age-standardized mortality rate per region-period.
    """
    if standard_pop is None:
        standard_pop = EUROPEAN_STANDARD_POPULATION

    total_standard = sum(standard_pop.values())

    # Compute age-specific rates and apply standard weights
    standardized_rate = pd.Series(0.0, index=deaths_by_age.index)

    for age_group, weight in standard_pop.items():
        if age_group in deaths_by_age.columns and age_group in population_by_age.columns:
            age_rate = deaths_by_age[age_group] / population_by_age[age_group]
            standardized_rate += age_rate * (weight / total_standard)

    return standardized_rate * per


def compute_excess_mortality(
    observed: pd.Series,
    expected: pd.Series,
) -> pd.DataFrame:
    """Compute excess mortality (absolute and relative).

    Parameters
    ----------
    observed : pd.Series
        Observed mortality counts or rates.
    expected : pd.Series
        Expected (baseline) mortality counts or rates.

    Returns
    -------
    pd.DataFrame
        With columns: excess_absolute, excess_relative_pct.
    """
    excess_abs = observed - expected
    excess_rel = ((observed - expected) / expected) * 100

    return pd.DataFrame(
        {
            "excess_absolute": excess_abs,
            "excess_relative_pct": excess_rel,
        }
    )


def compute_baseline_expected_mortality(
    mortality_panel: pd.DataFrame,
    baseline_years: tuple[int, int] = (2012, 2019),
) -> pd.DataFrame:
    """Compute expected (baseline) mortality from pre-pandemic years.

    Parameters
    ----------
    mortality_panel : pd.DataFrame
        Panel data with nuts2_code, year, and deaths columns.
    baseline_years : tuple of int
        Start and end year for the baseline period (default: 2012–2019,
        excluding COVID years).

    Returns
    -------
    pd.DataFrame
        Expected mortality per region (mean of baseline years).
    """
    baseline = mortality_panel[
        (mortality_panel["year"] >= baseline_years[0])
        & (mortality_panel["year"] <= baseline_years[1])
    ]

    expected = (
        baseline.groupby("nuts2_code")
        .agg(
            expected_deaths=("summer_deaths", "mean"),
            expected_deaths_std=("summer_deaths", "std"),
        )
        .reset_index()
    )

    logger.info(
        f"Computed baseline expected mortality from {baseline_years[0]}–{baseline_years[1]} "
        f"for {len(expected)} regions"
    )
    return expected
