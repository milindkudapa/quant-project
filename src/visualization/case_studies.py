"""
Case study comparison plots.

Generates paired regional comparison visualizations for the three
case study pairs defined in the project design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import load_config, get_path


def plot_case_study_pair(
    panel: pd.DataFrame,
    region_a: str,
    region_b: str,
    pair_name: str,
    output_path: Path | None = None,
    figsize: tuple[int, int] = (16, 14),
) -> None:
    """Plot a comprehensive comparison between two regions.

    Shows: mortality rates, heatwave days, RSVI, temperature, and
    the difference between regions over time.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset.
    region_a : str
        Name of the first region (high vulnerability).
    region_b : str
        Name of the second region (low vulnerability).
    pair_name : str
        Label for the comparison (e.g., "Economic Divide").
    output_path : Path, optional
        Save path.
    figsize : tuple
        Figure size.
    """
    name_col = "region_name" if "region_name" in panel.columns else "nuts2_code"

    # Get data for each region
    data_a = panel[panel[name_col] == region_a].sort_values("year")
    data_b = panel[panel[name_col] == region_b].sort_values("year")

    if data_a.empty or data_b.empty:
        logger.warning(f"Missing data for case study: {region_a} or {region_b}")
        return

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    colors = {"a": "#e74c3c", "b": "#2980b9"}

    # Panel 1: Mortality rate
    ax1 = fig.add_subplot(gs[0, 0])
    if "mortality_rate" in panel.columns:
        ax1.plot(data_a["year"], data_a["mortality_rate"], "o-",
                 color=colors["a"], label=region_a, linewidth=2)
        ax1.plot(data_b["year"], data_b["mortality_rate"], "s-",
                 color=colors["b"], label=region_b, linewidth=2)
        ax1.set_ylabel("Mortality Rate (per 100k)")
        ax1.set_title("Summer Mortality Rate")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Panel 2: Heatwave days
    ax2 = fig.add_subplot(gs[0, 1])
    if "hw_days" in panel.columns:
        width = 0.35
        x = data_a["year"].values
        ax2.bar(x - width / 2, data_a["hw_days"].values, width,
                color=colors["a"], label=region_a, alpha=0.8)
        ax2.bar(x + width / 2, data_b["hw_days"].values, width,
                color=colors["b"], label=region_b, alpha=0.8)
        ax2.set_ylabel("Heatwave Days")
        ax2.set_title("Heatwave Exposure")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Panel 3: Summer Tmax
    ax3 = fig.add_subplot(gs[1, 0])
    if "summer_tmax_mean" in panel.columns:
        ax3.plot(data_a["year"], data_a["summer_tmax_mean"], "o-",
                 color=colors["a"], label=region_a, linewidth=2)
        ax3.plot(data_b["year"], data_b["summer_tmax_mean"], "s-",
                 color=colors["b"], label=region_b, linewidth=2)
        ax3.set_ylabel("Mean Summer Tmax (°C)")
        ax3.set_title("Summer Temperature")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Panel 4: RSVI
    ax4 = fig.add_subplot(gs[1, 1])
    if "rsvi" in panel.columns:
        ax4.plot(data_a["year"], data_a["rsvi"], "o-",
                 color=colors["a"], label=region_a, linewidth=2)
        ax4.plot(data_b["year"], data_b["rsvi"], "s-",
                 color=colors["b"], label=region_b, linewidth=2)
        ax4.set_ylabel("RSVI")
        ax4.set_title("Social Vulnerability Index")
        ax4.set_ylim(0, 1)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Panel 5: Mortality difference (A - B)
    ax5 = fig.add_subplot(gs[2, :])
    if "mortality_rate" in panel.columns:
        merged = data_a[["year", "mortality_rate"]].merge(
            data_b[["year", "mortality_rate"]],
            on="year", suffixes=("_a", "_b"),
        )
        merged["diff"] = merged["mortality_rate_a"] - merged["mortality_rate_b"]

        colors_bar = ["#e74c3c" if d > 0 else "#2980b9" for d in merged["diff"]]
        ax5.bar(merged["year"], merged["diff"], color=colors_bar, alpha=0.8)
        ax5.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        ax5.set_ylabel(f"Mortality Difference\n({region_a} − {region_b})")
        ax5.set_title("Mortality Gap Over Time")
        ax5.set_xlabel("Year")
        ax5.grid(True, alpha=0.3)
        ax5.axvline(x=2022, color="darkred", linestyle="--", alpha=0.5, label="2022")
        ax5.legend()

    fig.suptitle(
        f"Case Study: {pair_name}\n{region_a} (high vulnerability) vs. {region_b} (low vulnerability)",
        fontsize=15, fontweight="bold", y=1.01,
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved case study plot → {output_path}")

    plt.close(fig)


def generate_all_case_studies(cfg: dict[str, Any] | None = None) -> None:
    """Generate plots for all configured case study pairs.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.
    """
    if cfg is None:
        cfg = load_config()

    from src.utils.io import load_dataframe

    panel_path = get_path(cfg, "processed_data") / "panel_dataset.csv"
    fig_dir = get_path(cfg, "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    panel = load_dataframe(panel_path)

    for pair in cfg["case_studies"]["pairs"]:
        name = pair["name"]
        high_v = pair["high_vulnerability"]
        low_v = pair["low_vulnerability"]

        safe_name = name.lower().replace(" ", "_")
        output_path = fig_dir / f"case_study_{safe_name}.png"

        plot_case_study_pair(
            panel,
            region_a=high_v,
            region_b=low_v,
            pair_name=name,
            output_path=output_path,
        )

    logger.success("All case study plots generated.")


if __name__ == "__main__":
    generate_all_case_studies()
