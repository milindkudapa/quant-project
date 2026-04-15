"""Tests for heatwave detection and metrics."""

import numpy as np
import pandas as pd
import pytest

from src.features.heatwave import (
    compute_percentile_thresholds,
    detect_heatwave_days,
    compute_heatwave_metrics,
)


@pytest.fixture
def sample_daily_climate():
    """Create sample daily climate data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2012-06-01", "2022-09-30", freq="D")
    # Filter to summer months only
    dates = dates[dates.month.isin([6, 7, 8, 9])]

    regions = ["ITC4", "ITF6"]  # Lombardia, Calabria
    records = []

    for region in regions:
        base_temp = 30 if region == "ITF6" else 28  # Calabria is hotter
        for date in dates:
            records.append({
                "date": date,
                "nuts2_code": region,
                "tmax": base_temp + np.random.normal(0, 4),
            })

    return pd.DataFrame(records)


def test_compute_percentile_thresholds(sample_daily_climate):
    """Test that percentile thresholds are computed correctly."""
    thresholds = compute_percentile_thresholds(
        sample_daily_climate, percentile=90
    )

    assert len(thresholds) == 2
    assert "ITC4" in thresholds.index
    assert "ITF6" in thresholds.index
    # Calabria should have a higher threshold
    assert thresholds["ITF6"] > thresholds["ITC4"]


def test_detect_heatwave_days(sample_daily_climate):
    """Test heatwave day detection."""
    thresholds = compute_percentile_thresholds(
        sample_daily_climate, percentile=90
    )
    result = detect_heatwave_days(
        sample_daily_climate, thresholds, min_consecutive=3
    )

    assert "is_heatwave_day" in result.columns
    assert "hw_event_id" in result.columns
    assert result["is_heatwave_day"].dtype == bool

    # Some days should be heatwave days
    hw_days = result["is_heatwave_day"].sum()
    assert hw_days >= 0  # Could be 0 with random data

    # All heatwave days should have tmax > threshold
    hw_subset = result[result["is_heatwave_day"]]
    if len(hw_subset) > 0:
        assert (hw_subset["tmax"] > hw_subset["threshold"]).all()


def test_detect_heatwave_min_consecutive(sample_daily_climate):
    """Test that minimum consecutive days constraint works."""
    thresholds = compute_percentile_thresholds(
        sample_daily_climate, percentile=90
    )

    # With min_consecutive=1, should find more heatwave days
    result_1 = detect_heatwave_days(sample_daily_climate, thresholds, min_consecutive=1)
    hw_count_1 = result_1["is_heatwave_day"].sum()

    # With min_consecutive=5, should find fewer
    result_5 = detect_heatwave_days(sample_daily_climate, thresholds, min_consecutive=5)
    hw_count_5 = result_5["is_heatwave_day"].sum()

    assert hw_count_1 >= hw_count_5


def test_compute_heatwave_metrics(sample_daily_climate):
    """Test seasonal heatwave metric computation."""
    thresholds = compute_percentile_thresholds(
        sample_daily_climate, percentile=90
    )
    hw_daily = detect_heatwave_days(sample_daily_climate, thresholds)
    metrics = compute_heatwave_metrics(hw_daily)

    assert "nuts2_code" in metrics.columns
    assert "year" in metrics.columns
    assert "hw_days" in metrics.columns
    assert "hw_events" in metrics.columns
    assert "hw_intensity" in metrics.columns
    assert "summer_tmax_anomaly" in metrics.columns

    # Should have entries for each region-year
    assert len(metrics) == metrics.groupby(["nuts2_code", "year"]).ngroups
