"""
Model diagnostics and robustness checks.

Provides tools for assessing panel regression model quality:
- Residual analysis
- Variance Inflation Factor (VIF) for multicollinearity
- Sensitivity analyses (excluding COVID years, alternative thresholds)
- Hausman test for fixed vs. random effects

Usage
-----
    python -m src.analysis.diagnostics
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects
from loguru import logger
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.utils.config import load_config, get_path
from src.utils.io import load_dataframe, save_dataframe


def compute_vif(panel: pd.DataFrame, exog_vars: list[str]) -> pd.DataFrame:
    """Compute Variance Inflation Factors for multicollinearity check.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data (flat, not multi-indexed).
    exog_vars : list of str
        Explanatory variable names.

    Returns
    -------
    pd.DataFrame
        VIF for each variable. VIF > 10 suggests problematic collinearity.
    """
    available = [v for v in exog_vars if v in panel.columns]
    data = panel[available].dropna()

    vif_data = pd.DataFrame({
        "variable": available,
        "VIF": [
            variance_inflation_factor(data.values, i)
            for i in range(len(available))
        ],
    })
    vif_data["problematic"] = vif_data["VIF"] > 10

    logger.info(f"VIF computed:\n{vif_data.to_string(index=False)}")
    return vif_data


def residual_diagnostics(
    results: Any,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run residual diagnostics on a fitted model.

    Parameters
    ----------
    results : PanelOLS results
        Fitted model results.
    output_dir : Path, optional
        Directory to save diagnostic plots.

    Returns
    -------
    dict
        Diagnostic statistics.
    """
    resids = results.resids

    diagnostics = {
        "mean_residual": float(resids.mean()),
        "std_residual": float(resids.std()),
        "skewness": float(stats.skew(resids.dropna())),
        "kurtosis": float(stats.kurtosis(resids.dropna())),
    }

    # Jarque-Bera test for normality
    jb_stat, jb_pval = stats.jarque_bera(resids.dropna())
    diagnostics["jarque_bera_stat"] = float(jb_stat)
    diagnostics["jarque_bera_pval"] = float(jb_pval)
    diagnostics["residuals_normal"] = jb_pval > 0.05

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Residual histogram
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(resids.dropna(), bins=30, edgecolor="white", alpha=0.8)
        axes[0].set_xlabel("Residuals")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Residual Distribution")
        axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5)

        # QQ plot
        stats.probplot(resids.dropna(), plot=axes[1])
        axes[1].set_title("Q-Q Plot")

        plt.tight_layout()
        fig.savefig(output_dir / "residual_diagnostics.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved residual diagnostics → {output_dir}")

    return diagnostics


def sensitivity_exclude_covid(
    panel: pd.DataFrame,
    dep_var: str = "mortality_rate",
) -> dict[str, Any]:
    """Run the H2 model excluding COVID year 2020.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data (multi-indexed or flat).
    dep_var : str
        Dependent variable.

    Returns
    -------
    dict
        Model results for the sensitivity analysis.
    """
    if "year" in panel.index.names:
        df = panel.copy()
        years = df.index.get_level_values("year")
        df = df[years != 2020]
    else:
        df = panel[panel["year"] != 2020].copy()
        df = df.set_index(["nuts2_code", "year"])

    exog_vars = ["hw_days", "rsvi", "hw_days_x_rsvi",
                 "summer_tmax_anomaly", "tmax_anomaly_x_rsvi"]
    available = [v for v in exog_vars if v in df.columns]

    model_data = df[[dep_var] + available].dropna()
    y = model_data[dep_var]
    x = model_data[available]

    model = PanelOLS(y, x, entity_effects=True, time_effects=True)
    results = model.fit(cov_type="clustered", cluster_entity=True)

    logger.info(f"Sensitivity (excl. 2020) — R²: {results.rsquared:.4f}")
    return {
        "model_name": "H2_no_covid",
        "results": results,
        "summary": str(results.summary),
    }


def hausman_test(
    panel: pd.DataFrame,
    dep_var: str,
    exog_vars: list[str],
) -> dict[str, float]:
    """Hausman test for fixed effects vs. random effects.

    Parameters
    ----------
    panel : pd.DataFrame
        Multi-indexed panel data.
    dep_var : str
        Dependent variable.
    exog_vars : list of str
        Explanatory variables.

    Returns
    -------
    dict
        Test statistic and p-value. If p < 0.05, fixed effects is preferred.
    """
    available = [v for v in exog_vars if v in panel.columns]
    model_data = panel[[dep_var] + available].dropna()
    y = model_data[dep_var]
    x = model_data[available]

    # Fixed effects
    fe_model = PanelOLS(y, x, entity_effects=True, time_effects=True)
    fe_results = fe_model.fit()

    # Random effects
    re_model = RandomEffects(y, x)
    re_results = re_model.fit()

    # Hausman statistic
    b_fe = fe_results.params
    b_re = re_results.params

    common_vars = b_fe.index.intersection(b_re.index)
    b_diff = b_fe[common_vars] - b_re[common_vars]

    # Covariance of the difference
    cov_diff = fe_results.cov - re_results.cov
    cov_diff = cov_diff.loc[common_vars, common_vars]

    try:
        hausman_stat = float(b_diff @ np.linalg.inv(cov_diff) @ b_diff)
        p_value = float(1 - stats.chi2.cdf(hausman_stat, df=len(common_vars)))
    except np.linalg.LinAlgError:
        logger.warning("Hausman test: singular covariance matrix")
        hausman_stat = np.nan
        p_value = np.nan

    result = {
        "hausman_stat": hausman_stat,
        "p_value": p_value,
        "df": len(common_vars),
        "prefer_fixed_effects": p_value < 0.05 if not np.isnan(p_value) else None,
    }

    logger.info(
        f"Hausman test: χ² = {hausman_stat:.2f}, p = {p_value:.4f} → "
        f"{'Fixed effects' if result['prefer_fixed_effects'] else 'Random effects'} preferred"
    )
    return result


def run_diagnostics(cfg: dict[str, Any] | None = None) -> None:
    """Run all diagnostic checks and save results.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.
    """
    if cfg is None:
        cfg = load_config()

    panel_path = get_path(cfg, "processed_data") / "panel_dataset.csv"
    fig_dir = get_path(cfg, "figures")
    table_dir = get_path(cfg, "tables")

    panel = load_dataframe(panel_path)

    # VIF
    exog_vars = [
        "hw_days", "rsvi", "hw_days_x_rsvi",
        "summer_tmax_anomaly", "tmax_anomaly_x_rsvi",
        "covid_period",
    ]
    vif = compute_vif(panel, exog_vars)
    save_dataframe(vif, table_dir / "vif_results.csv", index=False)

    logger.success("Diagnostics complete.")


if __name__ == "__main__":
    run_diagnostics()
