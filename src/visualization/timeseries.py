"""
Time series visualization.

Generates time series plots for mortality, heatwave days, and temperature
trends across the study period.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import load_config, get_path


def plot_national_trends(
    panel: pd.DataFrame,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (16, 12),
) -> None:
    """Plot national-level trends in mortality, heat, and vulnerability.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset.
    output_path : Path, optional
        Save path.
    figsize : tuple
        Figure size.
    """
    national = panel.groupby("year").agg({
        col: "mean" for col in panel.select_dtypes(include=[np.number]).columns
    }).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Mortality rate
    if "mortality_rate" in national.columns:
        axes[0, 0].plot(national["year"], national["mortality_rate"],
                        "o-", color="#e74c3c", linewidth=2)
        axes[0, 0].set_ylabel("Mortality Rate (per 100k)")
        axes[0, 0].set_title("A. Summer Mortality Rate")
        axes[0, 0].axvspan(2020, 2022, alpha=0.1, color="grey", label="COVID period")
        axes[0, 0].legend()

    # Heatwave days
    if "hw_days" in national.columns:
        axes[0, 1].bar(national["year"], national["hw_days"],
                       color="#e67e22", alpha=0.8)
        axes[0, 1].set_ylabel("Heatwave Days")
        axes[0, 1].set_title("B. Mean Heatwave Days")

    # Summer Tmax
    if "summer_tmax_mean" in national.columns:
        axes[1, 0].plot(national["year"], national["summer_tmax_mean"],
                        "s-", color="#2980b9", linewidth=2)
        axes[1, 0].set_ylabel("Summer Mean Tmax (°C)")
        axes[1, 0].set_title("C. Summer Temperature")

    # RSVI distribution
    if "rsvi" in panel.columns:
        rsvi_by_year = panel.groupby("year")["rsvi"].agg(["mean", "std"])
        axes[1, 1].fill_between(
            rsvi_by_year.index,
            rsvi_by_year["mean"] - rsvi_by_year["std"],
            rsvi_by_year["mean"] + rsvi_by_year["std"],
            alpha=0.3, color="#8e44ad",
        )
        axes[1, 1].plot(rsvi_by_year.index, rsvi_by_year["mean"],
                        "D-", color="#8e44ad", linewidth=2)
        axes[1, 1].set_ylabel("RSVI")
        axes[1, 1].set_title("D. Social Vulnerability (mean ± SD)")

    for ax in axes.flatten():
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3)

    fig.suptitle("National Trends (2012–2022)", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved national trends → {output_path}")

    plt.close(fig)


def plot_regional_comparison(
    panel: pd.DataFrame,
    variable: str,
    top_n: int = 5,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """Plot time series for the top and bottom N regions by a variable.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset.
    variable : str
        Variable to compare.
    top_n : int
        Number of top/bottom regions to highlight.
    output_path : Path, optional
        Save path.
    figsize : tuple
        Figure size.
    """
    if variable not in panel.columns:
        logger.warning(f"{variable} not in panel columns")
        return

    name_col = "region_name" if "region_name" in panel.columns else "nuts2_code"

    # Rank regions by mean of variable
    rankings = panel.groupby(name_col)[variable].mean().sort_values(ascending=False)
    top_regions = rankings.head(top_n).index.tolist()
    bottom_regions = rankings.tail(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot all regions in grey
    for region in panel[name_col].unique():
        subset = panel[panel[name_col] == region].sort_values("year")
        ax.plot(subset["year"], subset[variable], color="grey", alpha=0.15, linewidth=1)

    # Highlight top
    for region in top_regions:
        subset = panel[panel[name_col] == region].sort_values("year")
        ax.plot(subset["year"], subset[variable], linewidth=2, label=region, marker="o", markersize=4)

    # Highlight bottom
    for region in bottom_regions:
        subset = panel[panel[name_col] == region].sort_values("year")
        ax.plot(subset["year"], subset[variable], linewidth=2, label=region,
                marker="s", markersize=4, linestyle="--")

    ax.set_xlabel("Year")
    ax.set_ylabel(variable)
    ax.set_title(f"Regional Comparison — {variable}", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved regional comparison → {output_path}")

    plt.close(fig)
