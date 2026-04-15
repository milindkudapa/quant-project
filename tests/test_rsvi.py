"""Tests for RSVI construction."""

import numpy as np
import pandas as pd
import pytest

from src.features.rsvi import (
    percentile_rank_within_year,
    compute_sub_indices,
    compute_composite_rsvi,
)


@pytest.fixture
def sample_socioeconomic():
    """Create sample socioeconomic data for testing RSVI."""
    np.random.seed(42)
    regions = ["ITC4", "ITF6", "ITI4", "ITF4", "ITH4"]  # 5 regions
    years = [2020, 2021, 2022]

    records = []
    for region in regions:
        for year in years:
            records.append({
                "nuts2_code": region,
                "year": year,
                "pct_pop_65plus": np.random.uniform(15, 30),
                "pct_pop_75plus": np.random.uniform(8, 15),
                "pct_pop_80plus": np.random.uniform(4, 10),
                "poverty_rate_absolute": np.random.uniform(5, 25),
                "gdp_per_capita_inv": -np.random.uniform(20000, 40000),
                "disposable_income_inv": -np.random.uniform(15000, 30000),
                "population_density": np.random.uniform(50, 500),
                "urbanization_rate": np.random.uniform(30, 90),
            })

    return pd.DataFrame(records)


def test_percentile_rank_within_year(sample_socioeconomic):
    """Test percentile ranking."""
    indicators = ["pct_pop_65plus", "poverty_rate_absolute"]
    result = percentile_rank_within_year(sample_socioeconomic, indicators)

    assert "pct_pop_65plus_pctl" in result.columns
    assert "poverty_rate_absolute_pctl" in result.columns

    # Percentiles should be between 0 and 1
    for col in ["pct_pop_65plus_pctl", "poverty_rate_absolute_pctl"]:
        assert result[col].min() >= 0
        assert result[col].max() <= 1


def test_compute_sub_indices(sample_socioeconomic):
    """Test sub-index computation."""
    indicators = ["pct_pop_65plus", "pct_pop_75plus"]
    ranked = percentile_rank_within_year(sample_socioeconomic, indicators)

    sub_config = {"demographic": ["pct_pop_65plus", "pct_pop_75plus"]}
    result = compute_sub_indices(ranked, sub_config)

    assert "subidx_demographic" in result.columns
    assert result["subidx_demographic"].min() >= 0
    assert result["subidx_demographic"].max() <= 1


def test_compute_composite_rsvi(sample_socioeconomic):
    """Test composite RSVI computation."""
    all_indicators = [
        "pct_pop_65plus", "pct_pop_75plus", "pct_pop_80plus",
        "poverty_rate_absolute", "gdp_per_capita_inv",
        "population_density",
    ]
    ranked = percentile_rank_within_year(sample_socioeconomic, all_indicators)

    sub_config = {
        "demographic": ["pct_pop_65plus", "pct_pop_75plus", "pct_pop_80plus"],
        "economic": ["poverty_rate_absolute", "gdp_per_capita_inv"],
        "urban": ["population_density"],
    }
    with_subs = compute_sub_indices(ranked, sub_config)
    result = compute_composite_rsvi(with_subs, list(sub_config.keys()))

    assert "rsvi" in result.columns
    assert result["rsvi"].min() >= 0
    assert result["rsvi"].max() <= 1

    # Every region-year should have an RSVI
    assert not result["rsvi"].isna().any()


def test_rsvi_varies_across_regions(sample_socioeconomic):
    """Test that RSVI actually varies across regions."""
    all_indicators = [
        "pct_pop_65plus", "poverty_rate_absolute", "population_density",
    ]
    ranked = percentile_rank_within_year(sample_socioeconomic, all_indicators)

    sub_config = {
        "demographic": ["pct_pop_65plus"],
        "economic": ["poverty_rate_absolute"],
        "urban": ["population_density"],
    }
    with_subs = compute_sub_indices(ranked, sub_config)
    result = compute_composite_rsvi(with_subs, list(sub_config.keys()))

    # RSVI should vary across regions within each year
    for year in result["year"].unique():
        year_data = result[result["year"] == year]
        assert year_data["rsvi"].std() > 0, f"No RSVI variation in {year}"
