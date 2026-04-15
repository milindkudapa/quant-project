"""Tests for panel regression models."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.panel_regression import (
    prepare_panel_index,
    extract_results_table,
)


@pytest.fixture
def sample_panel():
    """Create a sample panel dataset for testing."""
    np.random.seed(42)
    regions = ["ITC4", "ITF6", "ITI4", "ITF4", "ITH4"]
    years = list(range(2012, 2023))

    records = []
    for region in regions:
        base_mort = 300 + np.random.normal(0, 20)
        base_rsvi = np.random.uniform(0.2, 0.8)

        for year in years:
            hw_days = max(0, int(np.random.normal(10, 5)))
            rsvi = base_rsvi + np.random.normal(0, 0.05)
            rsvi = np.clip(rsvi, 0, 1)

            # Mortality increases with heat and vulnerability
            mortality_rate = (
                base_mort
                + 2 * hw_days
                + 50 * rsvi
                + 3 * hw_days * rsvi
                + np.random.normal(0, 10)
            )

            records.append({
                "nuts2_code": region,
                "year": year,
                "mortality_rate": mortality_rate,
                "hw_days": hw_days,
                "summer_tmax_anomaly": np.random.normal(0, 1.5),
                "rsvi": rsvi,
                "hw_days_x_rsvi": hw_days * rsvi,
                "tmax_anomaly_x_rsvi": np.random.normal(0, 1.5) * rsvi,
                "covid_period": 1 if year >= 2020 else 0,
                "d2022": 1 if year == 2022 else 0,
                "hw_days_x_rsvi_x_d2022": hw_days * rsvi * (1 if year == 2022 else 0),
                "tmax_anomaly_x_rsvi_x_d2022": (
                    np.random.normal(0, 1.5) * rsvi * (1 if year == 2022 else 0)
                ),
            })

    return pd.DataFrame(records)


def test_prepare_panel_index(sample_panel):
    """Test that panel indexing is set correctly."""
    indexed = prepare_panel_index(sample_panel)
    assert indexed.index.names == ["nuts2_code", "year"]
    assert len(indexed) == len(sample_panel)


def test_extract_results_table():
    """Test results table extraction with mock data."""
    # Create mock results
    mock_results = [
        {
            "model_name": "test_model",
            "results": None,
            "summary": "No data",
        }
    ]

    table = extract_results_table(mock_results)
    assert isinstance(table, pd.DataFrame)
    # With None results, should be empty
    assert len(table) == 0
