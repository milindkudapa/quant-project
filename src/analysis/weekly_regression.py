"""
Weekly panel regressions with non-linear heat terms and distributed lags.

Three hypotheses, each tested on the region-week panel (summer weeks, W22–W39,
2012–2022, 20 NUTS-2 regions):

H1 (weekly base)
    mortality_week = f(heat) + region FE + year FE + week-of-year FE
    Non-linear heat: hw_days + tmax_anomaly + tmax_anomaly² + above-p95 indicator
    Distributed lag (lag0, lag1, lag2) on hw_days and tmax_anomaly.

H2 (weekly interaction)
    adds rsvi main effect and heat × rsvi interactions (lag0 + lag1).

H3 (weekly 2022 amplification)
    adds heat × rsvi × d2022 triple interactions.

Standard errors are clustered by region. We also fit a pooled (no region FE)
variant that keeps the between-region RSVI variation identified — region FE
absorbs most of the cross-sectional RSVI signal, so the pooled model is a
useful complement.

Robustness:
    - Exclude 2020 (COVID-contaminated weeks)
    - Exclude 2020 & 2021

Usage
-----
    python -m src.analysis.weekly_regression
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS
from loguru import logger

from src.utils.config import load_config, get_path
from src.utils.io import load_dataframe, save_dataframe


BASE_HEAT = [
    "hw_days_week",
    "hw_days_week_lag1",
    "hw_days_week_lag2",
    "tmax_anomaly_week",
    "tmax_anomaly_week_sq",
    "tmax_anomaly_week_lag1",
    "hot_week_p95",
]
RSVI_INTERACTIONS = [
    "hw_days_x_rsvi",
    "tmax_anom_x_rsvi",
    "tmax_anom_sq_x_rsvi",
    "p95_x_rsvi",
    "hw_days_week_lag1_x_rsvi",
    "hw_days_week_lag2_x_rsvi",
]
TRIPLE_INTERACTIONS = [
    "hw_days_x_rsvi_x_d2022",
    "tmax_anom_x_rsvi_x_d2022",
    "p95_x_rsvi_x_d2022",
]


def _add_fe_dummies(x: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Add year and week-of-year dummies (drop first level of each)."""
    year_d = pd.get_dummies(panel["iso_year"], prefix="yr", drop_first=True).astype(
        float
    )
    week_d = pd.get_dummies(panel["iso_week"], prefix="wk", drop_first=True).astype(
        float
    )
    year_d.index = x.index
    week_d.index = x.index
    return pd.concat([x, year_d, week_d], axis=1)


def _prepare(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p = p.set_index(["nuts2_code", "week_id"])
    return p


def _fit(
    panel: pd.DataFrame,
    regressors: list[str],
    dep_var: str,
    entity_fe: bool,
) -> dict[str, Any]:
    # Year FE absorb any year-level COVID dummy, so we do not add covid_period
    # as a separate regressor here.
    df = _prepare(panel)
    cols = [c for c in regressors if c in df.columns]
    model_data = df[[dep_var] + cols].dropna()
    y = model_data[dep_var]
    x = model_data[cols].copy()
    # Add year and week-of-year dummies based on the rows surviving .dropna()
    panel_aligned = panel.set_index(["nuts2_code", "week_id"]).loc[model_data.index]
    x = _add_fe_dummies(x, panel_aligned)
    # Drop zero-variance columns (if any dummy is entirely 0/1 after subsetting)
    x = x.loc[:, x.nunique() > 1]

    if entity_fe:
        model = PanelOLS(y, x, entity_effects=True, drop_absorbed=True)
    else:
        model = PooledOLS(y, x)
    results = model.fit(cov_type="clustered", cluster_entity=True)
    return {"results": results, "n": int(results.nobs), "r2": float(results.rsquared)}


def run_weekly_models(
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()

    panel_path = get_path(cfg, "processed_data") / "weekly_panel_dataset.parquet"
    panel = load_dataframe(panel_path)

    dep_var = "log_mortality_rate_week"

    model_specs = {
        "H1_weekly_FE": (BASE_HEAT, True, panel),
        "H2_weekly_FE": (BASE_HEAT + ["rsvi"] + RSVI_INTERACTIONS, True, panel),
        "H3_weekly_FE": (
            BASE_HEAT + ["rsvi"] + RSVI_INTERACTIONS + TRIPLE_INTERACTIONS,
            True,
            panel,
        ),
        # Pooled variants — keep RSVI between-region variation identified
        "H2_weekly_pooled": (BASE_HEAT + ["rsvi"] + RSVI_INTERACTIONS, False, panel),
        "H3_weekly_pooled": (
            BASE_HEAT + ["rsvi"] + RSVI_INTERACTIONS + TRIPLE_INTERACTIONS,
            False,
            panel,
        ),
        # Robustness: drop 2020 (COVID year)
        "H3_weekly_FE_excl2020": (
            BASE_HEAT + ["rsvi"] + RSVI_INTERACTIONS + TRIPLE_INTERACTIONS,
            True,
            panel[panel["iso_year"] != 2020].copy(),
        ),
        # Robustness: drop 2020 & 2021
        "H3_weekly_FE_excl_covid": (
            BASE_HEAT + ["rsvi"] + RSVI_INTERACTIONS + TRIPLE_INTERACTIONS,
            True,
            panel[~panel["iso_year"].isin([2020, 2021])].copy(),
        ),
    }

    rows = []
    summaries: dict[str, str] = {}
    for name, (regs, fe, data) in model_specs.items():
        try:
            res = _fit(data, regs, dep_var=dep_var, entity_fe=fe)
        except Exception as exc:
            logger.error(f"{name} failed: {exc}")
            continue
        r = res["results"]
        summaries[name] = str(r.summary)
        logger.info(
            f"{name}: N={res['n']}, within-R²={res['r2']:.4f}"
        )
        for var in r.params.index:
            # Skip FE dummies from the main results table to keep it readable
            if var.startswith("yr_") or var.startswith("wk_"):
                continue
            rows.append(
                {
                    "model": name,
                    "variable": var,
                    "coefficient": float(r.params[var]),
                    "std_error": float(r.std_errors[var]),
                    "t_stat": float(r.tstats[var]),
                    "p_value": float(r.pvalues[var]),
                    "significant_5pct": float(r.pvalues[var]) < 0.05,
                    "significant_1pct": float(r.pvalues[var]) < 0.01,
                    "N": res["n"],
                    "within_R2": res["r2"],
                }
            )

    out = pd.DataFrame(rows)
    table_dir = get_path(cfg, "tables")
    table_dir.mkdir(parents=True, exist_ok=True)
    save_dataframe(out, table_dir / "regression_results_weekly.csv", index=False)

    for name, text in summaries.items():
        (table_dir / f"model_{name}_summary.txt").write_text(text)

    logger.success(
        f"Weekly regressions complete. {len(model_specs)} models, "
        f"{len(out)} coefficient rows saved."
    )
    return out


if __name__ == "__main__":
    run_weekly_models()
