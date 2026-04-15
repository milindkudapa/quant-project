"""
Choropleth and spatial maps for Italian NUTS-2 regions.

Generates maps of mortality rates, heatwave exposure, and RSVI
across Italian regions.

Usage
-----
    python -m src.visualization.maps
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from loguru import logger

from src.utils.config import load_config, get_path


def plot_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    cmap: str = "YlOrRd",
    legend_label: str = "",
    output_path: Path | None = None,
    figsize: tuple[int, int] = (10, 12),
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Plot a choropleth map of Italian NUTS-2 regions.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with NUTS-2 geometries and data columns.
    column : str
        Column to visualize.
    title : str
        Map title.
    cmap : str
        Matplotlib colormap name.
    legend_label : str
        Label for the colorbar.
    output_path : Path, optional
        Path to save the figure.
    figsize : tuple
        Figure size.
    vmin, vmax : float, optional
        Colorbar range limits.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    gdf.plot(
        column=column,
        cmap=cmap,
        linewidth=0.5,
        edgecolor="0.3",
        legend=True,
        legend_kwds={
            "label": legend_label or column,
            "orientation": "horizontal",
            "shrink": 0.6,
            "pad": 0.02,
        },
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        missing_kwds={"color": "lightgrey", "hatch": "///", "label": "No data"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_axis_off()
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved choropleth → {output_path}")

    plt.close(fig)


def plot_multi_year_choropleth(
    gdf: gpd.GeoDataFrame,
    column: str,
    years: list[int],
    title: str,
    cmap: str = "YlOrRd",
    output_path: Path | None = None,
) -> None:
    """Plot choropleth maps for multiple years in a grid.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with year column and data.
    column : str
        Column to visualize.
    years : list of int
        Years to plot.
    title : str
        Overall title.
    cmap : str
        Colormap.
    output_path : Path, optional
        Save path.
    """
    n_years = len(years)
    n_cols = min(4, n_years)
    n_rows = (n_years + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows))
    axes = np.array(axes).flatten() if n_years > 1 else [axes]

    # Shared color scale
    vmin = gdf[column].min()
    vmax = gdf[column].max()

    for i, year in enumerate(years):
        ax = axes[i]
        year_data = gdf[gdf["year"] == year]

        if year_data.empty:
            ax.set_title(f"{year} (no data)")
            ax.set_axis_off()
            continue

        year_data.plot(
            column=column,
            cmap=cmap,
            linewidth=0.5,
            edgecolor="0.3",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            missing_kwds={"color": "lightgrey"},
        )
        ax.set_title(str(year), fontsize=12, fontweight="bold")
        ax.set_axis_off()

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()

    # Shared colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", shrink=0.6, pad=0.05)
    cbar.set_label(column)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved multi-year choropleth → {output_path}")

    plt.close(fig)


def merge_panel_with_geometry(
    panel: pd.DataFrame,
    nuts2_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Merge panel data with NUTS-2 geometries for mapping.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel dataset with nuts2_code column.
    nuts2_gdf : gpd.GeoDataFrame
        NUTS-2 boundaries.

    Returns
    -------
    gpd.GeoDataFrame
        Merged GeoDataFrame.
    """
    merged = nuts2_gdf.merge(panel, left_on="NUTS_ID", right_on="nuts2_code", how="right")
    return gpd.GeoDataFrame(merged, geometry="geometry")


def generate_all_maps(cfg: dict[str, Any] | None = None) -> None:
    """Generate all project maps.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.
    """
    if cfg is None:
        cfg = load_config()

    from src.utils.io import load_dataframe
    from src.data.nuts2_boundaries import load_italy_nuts2

    panel_path = get_path(cfg, "processed_data") / "panel_dataset.csv"
    boundaries_dir = get_path(cfg, "raw_data") / "boundaries"
    fig_dir = get_path(cfg, "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    panel = load_dataframe(panel_path)
    nuts2_gdf = load_italy_nuts2(boundaries_dir)
    gdf = merge_panel_with_geometry(panel, nuts2_gdf)

    # 2022 snapshot maps
    gdf_2022 = gdf[gdf["year"] == 2022]

    if "hw_days" in gdf_2022.columns:
        plot_choropleth(
            gdf_2022, "hw_days",
            title="Heatwave Days — Summer 2022",
            cmap="YlOrRd",
            legend_label="Number of Heatwave Days",
            output_path=fig_dir / "heatwave_map_2022.png",
        )

    if "rsvi" in gdf_2022.columns:
        plot_choropleth(
            gdf_2022, "rsvi",
            title="Regional Social Vulnerability Index — 2022",
            cmap="PuRd",
            legend_label="RSVI (0 = Low, 1 = High)",
            output_path=fig_dir / "rsvi_map_2022.png",
        )

    if "mortality_rate" in gdf_2022.columns:
        plot_choropleth(
            gdf_2022, "mortality_rate",
            title="Summer Mortality Rate — 2022",
            cmap="Reds",
            legend_label="Deaths per 100,000",
            output_path=fig_dir / "mortality_map_2022.png",
        )

    # Multi-year timeseries maps
    years = sorted(panel["year"].unique())
    if "hw_days" in gdf.columns:
        plot_multi_year_choropleth(
            gdf, "hw_days", years,
            title="Heatwave Days Across Italy (2012–2022)",
            output_path=fig_dir / "heatwave_map_all_years.png",
        )

    logger.success("All maps generated.")


if __name__ == "__main__":
    generate_all_maps()
