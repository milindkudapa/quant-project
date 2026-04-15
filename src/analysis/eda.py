"""
Exploratory data analysis.

Generates summary statistics, correlation matrices, time series plots,
and distribution diagnostics for the panel dataset.

Usage
-----
    python -m src.analysis.eda
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from src.utils.config import load_config, get_path
from src.utils.io import load_dataframe, save_dataframe


def summary_statistics(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for all numeric variables.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset.

    Returns
    -------
    pd.DataFrame
        Summary statistics table.
    """
    numeric_cols = panel.select_dtypes(include=[np.number]).columns
    stats = panel[numeric_cols].describe().T
    stats["missing"] = panel[numeric_cols].isna().sum()
    stats["missing_pct"] = (stats["missing"] / len(panel)) * 100

    logger.info(f"Computed summary statistics for {len(stats)} variables")
    return stats


def summary_by_region(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by region.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset.

    Returns
    -------
    pd.DataFrame
        Regional summary statistics.
    """
    key_vars = [
        "summer_deaths", "mortality_rate", "hw_days", "hw_events",
        "hw_intensity", "summer_tmax_mean", "summer_tmax_anomaly",
        "rsvi",
    ]
    available = [v for v in key_vars if v in panel.columns]

    regional = (
        panel.groupby("nuts2_code")[available]
        .agg(["mean", "std", "min", "max"])
    )
    return regional


def correlation_matrix(
    panel: pd.DataFrame,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (14, 10),
) -> pd.DataFrame:
    """Compute and plot correlation matrix for key variables.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset.
    output_path : Path, optional
        Path to save the figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    key_vars = [
        "mortality_rate", "hw_days", "hw_intensity", "hw_max_duration",
        "summer_tmax_mean", "summer_tmax_anomaly", "rsvi",
    ]
    available = [v for v in key_vars if v in panel.columns]

    corr = panel[available].corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Matrix — Key Panel Variables", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved correlation matrix → {output_path}")

    plt.close(fig)
    return corr


def plot_heatwave_timeseries(
    panel: pd.DataFrame,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (16, 8),
) -> None:
    """Plot heatwave days over time by region.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset.
    output_path : Path, optional
        Path to save the figure.
    figsize : tuple
        Figure size.
    """
    if "hw_days" not in panel.columns:
        logger.warning("hw_days column not found, skipping timeseries plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel A: National average heatwave days per year
    national = panel.groupby("year")["hw_days"].agg(["mean", "std"])
    axes[0].bar(national.index, national["mean"], color="#e74c3c", alpha=0.8)
    axes[0].errorbar(national.index, national["mean"], yerr=national["std"],
                     fmt="none", color="black", capsize=3)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Heatwave Days (mean ± SD)")
    axes[0].set_title("A. National Average Heatwave Days")
    axes[0].axvline(x=2022, color="darkred", linestyle="--", alpha=0.5, label="2022")
    axes[0].legend()

    # Panel B: Regional variation
    name_col = "region_name" if "region_name" in panel.columns else "nuts2_code"
    for region in panel[name_col].unique():
        subset = panel[panel[name_col] == region].sort_values("year")
        axes[1].plot(subset["year"], subset["hw_days"], alpha=0.4, linewidth=1)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Heatwave Days")
    axes[1].set_title("B. Heatwave Days by Region")

    plt.suptitle("Heatwave Exposure (2012–2022)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved heatwave timeseries → {output_path}")

    plt.close(fig)


def plot_mortality_vs_heat(
    panel: pd.DataFrame,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """Scatter plot of mortality rate vs. heatwave days, colored by RSVI.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset.
    output_path : Path, optional
        Path to save the figure.
    figsize : tuple
        Figure size.
    """
    required = ["mortality_rate", "hw_days", "rsvi"]
    if not all(col in panel.columns for col in required):
        logger.warning(f"Missing columns for scatter plot: {required}")
        return

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        panel["hw_days"],
        panel["mortality_rate"],
        c=panel["rsvi"],
        cmap="YlOrRd",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
        s=60,
    )
    plt.colorbar(scatter, ax=ax, label="RSVI (Social Vulnerability)")
    ax.set_xlabel("Heatwave Days (summer)")
    ax.set_ylabel("Mortality Rate (per 100,000)")
    ax.set_title(
        "Mortality Rate vs. Heatwave Exposure\nColored by Social Vulnerability (RSVI)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved mortality vs heat scatter → {output_path}")

    plt.close(fig)


def run_eda(cfg: dict[str, Any] | None = None) -> None:
    """Run the full EDA pipeline.

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

    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    panel = load_dataframe(panel_path)

    # Summary stats
    stats = summary_statistics(panel)
    save_dataframe(stats, table_dir / "summary_stats.csv")

    regional = summary_by_region(panel)
    save_dataframe(regional, table_dir / "summary_by_region.csv")

    # Plots
    correlation_matrix(panel, output_path=fig_dir / "correlation_matrix.png")
    plot_heatwave_timeseries(panel, output_path=fig_dir / "heatwave_timeseries.png")
    plot_mortality_vs_heat(panel, output_path=fig_dir / "mortality_vs_heat.png")

    logger.success("EDA complete. Outputs saved to outputs/figures/ and outputs/tables/")


if __name__ == "__main__":
    run_eda()
