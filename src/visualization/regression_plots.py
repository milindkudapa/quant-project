"""
Regression result visualization.

Coefficient plots, interaction effect plots, and marginal effects
for the panel regression models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


def plot_coefficient_comparison(
    results_table: pd.DataFrame,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """Plot coefficient estimates across models with confidence intervals.

    Parameters
    ----------
    results_table : pd.DataFrame
        Regression results with columns: model, variable, coefficient,
        std_error, p_value.
    output_path : Path, optional
        Save path.
    figsize : tuple
        Figure size.
    """
    # Get unique variables (excluding intercept-like terms)
    key_vars = [
        "hw_days", "summer_tmax_anomaly", "rsvi",
        "hw_days_x_rsvi", "tmax_anomaly_x_rsvi",
        "hw_days_x_rsvi_x_d2022", "tmax_anomaly_x_rsvi_x_d2022",
    ]
    plot_data = results_table[results_table["variable"].isin(key_vars)].copy()

    if plot_data.empty:
        logger.warning("No relevant variables found for coefficient plot")
        return

    models = plot_data["model"].unique()
    variables = plot_data["variable"].unique()

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(len(variables))
    model_offsets = np.linspace(-0.2, 0.2, len(models))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    for i, model_name in enumerate(models):
        model_data = plot_data[plot_data["model"] == model_name]

        for j, var in enumerate(variables):
            row = model_data[model_data["variable"] == var]
            if row.empty:
                continue

            coef = row["coefficient"].values[0]
            se = row["std_error"].values[0]
            pval = row["p_value"].values[0]

            # 95% CI
            ci_low = coef - 1.96 * se
            ci_high = coef + 1.96 * se

            marker = "o" if pval < 0.05 else "x"
            ax.errorbar(
                coef, j + model_offsets[i],
                xerr=[[coef - ci_low], [ci_high - coef]],
                fmt=marker, color=colors[i], capsize=4,
                markersize=8, linewidth=1.5,
                label=model_name if j == 0 else None,
            )

    ax.axvline(x=0, color="grey", linestyle="--", alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(variables, fontsize=10)
    ax.set_xlabel("Coefficient Estimate (95% CI)", fontsize=12)
    ax.set_title("Panel Regression Coefficients — Model Comparison",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Model", loc="best")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved coefficient plot → {output_path}")

    plt.close(fig)


def plot_interaction_effect(
    panel: pd.DataFrame,
    results: Any,
    heat_var: str = "hw_days",
    output_path: Path | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """Plot the marginal effect of heat on mortality at different RSVI levels.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset.
    results : PanelOLS results
        Fitted H2 model results.
    heat_var : str
        Heat exposure variable name.
    output_path : Path, optional
        Save path.
    figsize : tuple
        Figure size.
    """
    if results is None:
        logger.warning("No model results provided for interaction plot")
        return

    params = results.params

    # Marginal effect of heat at different RSVI levels
    # ∂Mortality/∂Heat = β₁ + β₂ × RSVI
    beta_heat = params.get(heat_var, 0)
    beta_interaction = params.get(f"{heat_var}_x_rsvi", 0)

    rsvi_range = np.linspace(0, 1, 100)
    marginal_effect = beta_heat + beta_interaction * rsvi_range

    # Confidence band (approximate)
    se_heat = results.std_errors.get(heat_var, 0)
    se_interaction = results.std_errors.get(f"{heat_var}_x_rsvi", 0)

    # Simple variance propagation
    se_marginal = np.sqrt(se_heat**2 + (rsvi_range * se_interaction)**2)
    ci_upper = marginal_effect + 1.96 * se_marginal
    ci_lower = marginal_effect - 1.96 * se_marginal

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(rsvi_range, marginal_effect, color="#e74c3c", linewidth=2,
            label="Marginal effect of heat")
    ax.fill_between(rsvi_range, ci_lower, ci_upper,
                    color="#e74c3c", alpha=0.15, label="95% CI")
    ax.axhline(y=0, color="grey", linestyle="--", alpha=0.5)

    # Mark RSVI percentiles
    if "rsvi" in panel.columns:
        for q, label in [(0.25, "25th pctl"), (0.50, "Median"), (0.75, "75th pctl")]:
            rsvi_val = panel["rsvi"].quantile(q)
            me_val = beta_heat + beta_interaction * rsvi_val
            ax.plot(rsvi_val, me_val, "D", color="#2c3e50", markersize=10, zorder=5)
            ax.annotate(f"{label}\n(RSVI={rsvi_val:.2f})",
                        (rsvi_val, me_val), textcoords="offset points",
                        xytext=(10, 10), fontsize=9)

    ax.set_xlabel("RSVI (Social Vulnerability)", fontsize=12)
    ax.set_ylabel(f"Marginal Effect of {heat_var} on Mortality", fontsize=12)
    ax.set_title(
        "How Social Vulnerability Modifies the Heat-Mortality Relationship",
        fontsize=14, fontweight="bold",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved interaction effect plot → {output_path}")

    plt.close(fig)
