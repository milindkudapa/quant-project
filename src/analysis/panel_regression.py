"""
Fixed-effects panel regression models.

Implements the three hypothesis tests:
- H1: Heat exposure → mortality (base model)
- H2: Heat × RSVI interaction (vulnerability moderation)
- H3: Heat × RSVI × 2022 (2022 amplification of inequality)

Uses the `linearmodels` package for panel data estimation with
entity (region) and time (year) fixed effects.

Usage
-----
    python -m src.analysis.panel_regression
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from loguru import logger

from src.utils.config import load_config, get_path
from src.utils.io import load_dataframe, save_dataframe


def prepare_panel_index(panel: pd.DataFrame) -> pd.DataFrame:
    """Set the multi-index required by linearmodels.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset with nuts2_code and year columns.

    Returns
    -------
    pd.DataFrame
        Panel with (nuts2_code, year) MultiIndex.
    """
    df = panel.copy()
    df = df.set_index(["nuts2_code", "year"])
    return df


def run_model_h1(
    panel: pd.DataFrame,
    dep_var: str = "mortality_rate",
    cluster: str | None = "nuts2_code",
) -> dict[str, Any]:
    """Run H1 model: Heat exposure → mortality.

    Model: mortality_rt = β₁·hw_days + β₂·summer_tmax_anomaly
           + β₃·covid_period + α_r + γ_t + ε_rt

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset with MultiIndex (nuts2_code, year).
    dep_var : str
        Dependent variable name.
    cluster : str, optional
        Variable to cluster standard errors by.

    Returns
    -------
    dict
        Model results with keys: model_name, results, summary.
    """
    exog_vars = []
    for var in ["hw_days", "summer_tmax_anomaly", "covid_period"]:
        if var in panel.columns:
            exog_vars.append(var)

    if not exog_vars or dep_var not in panel.columns:
        logger.warning(f"Missing variables for H1 model. Need: {dep_var}, {exog_vars}")
        return {"model_name": "H1", "results": None, "summary": "Insufficient data"}

    # Drop rows with missing values
    model_data = panel[[dep_var] + exog_vars].dropna()

    y = model_data[dep_var]
    x = model_data[exog_vars]

    model = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True)

    if cluster:
        results = model.fit(cov_type="clustered", cluster_entity=True)
    else:
        results = model.fit(cov_type="robust")

    logger.info(f"H1 Model — R²: {results.rsquared:.4f}")
    logger.info(f"H1 Model — Coefficients:\n{results.params}")

    return {
        "model_name": "H1_base",
        "results": results,
        "summary": str(results.summary),
    }


def run_model_h2(
    panel: pd.DataFrame,
    dep_var: str = "mortality_rate",
    cluster: str | None = "nuts2_code",
) -> dict[str, Any]:
    """Run H2 model: Heat × RSVI interaction.

    Model: mortality_rt = β₁·hw_days + β₂·rsvi + β₃·(hw_days × rsvi)
           + β₄·summer_tmax_anomaly + β₅·(tmax_anomaly × rsvi)
           + β₆·covid_period + α_r + γ_t + ε_rt

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset with MultiIndex.
    dep_var : str
        Dependent variable name.
    cluster : str, optional
        Clustering variable.

    Returns
    -------
    dict
        Model results.
    """
    exog_vars = []
    candidates = [
        "hw_days", "rsvi", "hw_days_x_rsvi",
        "summer_tmax_anomaly", "tmax_anomaly_x_rsvi",
        "covid_period",
    ]
    for var in candidates:
        if var in panel.columns:
            exog_vars.append(var)

    if not exog_vars or dep_var not in panel.columns:
        logger.warning(f"Missing variables for H2 model.")
        return {"model_name": "H2", "results": None, "summary": "Insufficient data"}

    model_data = panel[[dep_var] + exog_vars].dropna()
    y = model_data[dep_var]
    x = model_data[exog_vars]

    model = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True)

    if cluster:
        results = model.fit(cov_type="clustered", cluster_entity=True)
    else:
        results = model.fit(cov_type="robust")

    logger.info(f"H2 Model — R²: {results.rsquared:.4f}")
    logger.info(f"H2 Model — Key interaction (hw_days × rsvi): "
                f"{results.params.get('hw_days_x_rsvi', 'N/A')}")

    return {
        "model_name": "H2_interaction",
        "results": results,
        "summary": str(results.summary),
    }


def run_model_h3(
    panel: pd.DataFrame,
    dep_var: str = "mortality_rate",
    cluster: str | None = "nuts2_code",
) -> dict[str, Any]:
    """Run H3 model: 2022 amplification of inequality.

    Model: mortality_rt = ... + β·(hw_days × rsvi × D2022) + α_r + γ_t + ε_rt

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset with MultiIndex.
    dep_var : str
        Dependent variable name.
    cluster : str, optional
        Clustering variable.

    Returns
    -------
    dict
        Model results.
    """
    exog_vars = []
    candidates = [
        "hw_days", "rsvi", "hw_days_x_rsvi",
        "summer_tmax_anomaly", "tmax_anomaly_x_rsvi",
        "d2022", "hw_days_x_rsvi_x_d2022", "tmax_anomaly_x_rsvi_x_d2022",
        "covid_period",
    ]
    for var in candidates:
        if var in panel.columns:
            exog_vars.append(var)

    if not exog_vars or dep_var not in panel.columns:
        logger.warning(f"Missing variables for H3 model.")
        return {"model_name": "H3", "results": None, "summary": "Insufficient data"}

    model_data = panel[[dep_var] + exog_vars].dropna()
    y = model_data[dep_var]
    x = model_data[exog_vars]

    model = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True)

    if cluster:
        results = model.fit(cov_type="clustered", cluster_entity=True)
    else:
        results = model.fit(cov_type="robust")

    logger.info(f"H3 Model — R²: {results.rsquared:.4f}")
    logger.info(f"H3 Model — 2022 amplification (hw_days × rsvi × d2022): "
                f"{results.params.get('hw_days_x_rsvi_x_d2022', 'N/A')}")

    return {
        "model_name": "H3_2022_amplification",
        "results": results,
        "summary": str(results.summary),
    }


def extract_results_table(model_results: list[dict]) -> pd.DataFrame:
    """Extract a comparison table from multiple model results.

    Parameters
    ----------
    model_results : list of dict
        List of model result dicts from run_model_h* functions.

    Returns
    -------
    pd.DataFrame
        Comparison table with coefficients and significance.
    """
    rows = []

    for mr in model_results:
        if mr["results"] is None:
            continue

        res = mr["results"]
        for var_name in res.params.index:
            rows.append({
                "model": mr["model_name"],
                "variable": var_name,
                "coefficient": res.params[var_name],
                "std_error": res.std_errors[var_name],
                "t_stat": res.tstats[var_name],
                "p_value": res.pvalues[var_name],
                "significant_5pct": res.pvalues[var_name] < 0.05,
                "significant_1pct": res.pvalues[var_name] < 0.01,
            })

    return pd.DataFrame(rows)


def run_all_models(
    cfg: dict[str, Any] | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    """Run all three panel regression models and save results.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.

    Returns
    -------
    tuple
        (list of model result dicts, comparison table DataFrame)
    """
    if cfg is None:
        cfg = load_config()

    panel_path = get_path(cfg, "processed_data") / "panel_dataset.csv"
    table_dir = get_path(cfg, "tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    panel = load_dataframe(panel_path)
    panel = prepare_panel_index(panel)

    dep_var = cfg["regression"].get("dependent_var", "mortality_rate")

    # Run models
    results = [
        run_model_h1(panel, dep_var=dep_var),
        run_model_h2(panel, dep_var=dep_var),
        run_model_h3(panel, dep_var=dep_var),
    ]

    # Save results
    comparison = extract_results_table(results)
    save_dataframe(comparison, table_dir / "regression_results.csv", index=False)

    # Save individual model summaries
    for mr in results:
        if mr["summary"]:
            summary_path = table_dir / f"model_{mr['model_name']}_summary.txt"
            with open(summary_path, "w") as f:
                f.write(mr["summary"])
            logger.info(f"Saved model summary → {summary_path}")

    logger.success("All panel regression models complete.")
    return results, comparison


if __name__ == "__main__":
    run_all_models()
