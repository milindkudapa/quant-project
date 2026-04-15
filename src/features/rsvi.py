"""
Regional Social Vulnerability Index (RSVI) construction.

Implements the CDC/ATSDR SVI methodology adapted for Italian NUTS-2 regions:
1. Percentile-rank each indicator across regions (for each year)
2. Group into thematic sub-indices (Demographic, Economic, Urban/Infrastructure)
3. Average sub-index percentiles → composite RSVI score (0–1)

Usage
-----
    python -m src.features.rsvi
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from src.utils.config import load_config, get_path
from src.utils.io import load_dataframe, save_dataframe


def percentile_rank_within_year(
    df: pd.DataFrame,
    indicator_cols: list[str],
) -> pd.DataFrame:
    """Compute percentile ranks for each indicator within each year.

    Higher percentile = higher vulnerability. Indicators that need
    inversion (e.g., GDP per capita where lower = more vulnerable)
    should be pre-inverted before calling this function.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with columns: nuts2_code, year, and indicator columns.
    indicator_cols : list of str
        Column names of the vulnerability indicators.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with additional ``{col}_pctl`` columns.
    """
    result = df.copy()

    for col in indicator_cols:
        if col not in result.columns:
            logger.warning(f"Indicator '{col}' not found in data, skipping")
            continue

        pctl_col = f"{col}_pctl"
        result[pctl_col] = result.groupby("year")[col].transform(
            lambda x: x.rank(pct=True, na_option="keep")
        )

    logger.info(f"Computed percentile ranks for {len(indicator_cols)} indicators")
    return result


def compute_sub_indices(
    df: pd.DataFrame,
    sub_index_config: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute thematic sub-indices by averaging percentile-ranked indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Data with percentile-ranked indicators (``*_pctl`` columns).
    sub_index_config : dict
        Mapping of sub-index name → list of indicator names.
        E.g., ``{"demographic": ["pct_pop_65plus", "pct_pop_75plus"]}``.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional sub-index columns (``subidx_*``).
    """
    result = df.copy()

    for sub_name, indicators in sub_index_config.items():
        pctl_cols = [f"{ind}_pctl" for ind in indicators if f"{ind}_pctl" in result.columns]

        if not pctl_cols:
            logger.warning(f"No percentile columns found for sub-index '{sub_name}'")
            continue

        result[f"subidx_{sub_name}"] = result[pctl_cols].mean(axis=1)
        logger.info(
            f"Sub-index '{sub_name}': averaged {len(pctl_cols)} indicators "
            f"({', '.join(pctl_cols)})"
        )

    return result


def compute_composite_rsvi(
    df: pd.DataFrame,
    sub_index_names: list[str],
    method: str = "equal_weight_mean",
) -> pd.DataFrame:
    """Compute the composite RSVI from sub-indices.

    Parameters
    ----------
    df : pd.DataFrame
        Data with sub-index columns (``subidx_*``).
    sub_index_names : list of str
        Names of the sub-indices (matching ``subidx_{name}`` columns).
    method : str
        Aggregation method: "equal_weight_mean" (default) or "sum".

    Returns
    -------
    pd.DataFrame
        DataFrame with an ``rsvi`` column (0–1 scale).
    """
    result = df.copy()
    sub_cols = [f"subidx_{name}" for name in sub_index_names if f"subidx_{name}" in result.columns]

    if not sub_cols:
        raise ValueError("No sub-index columns found. Run compute_sub_indices() first.")

    if method == "equal_weight_mean":
        result["rsvi"] = result[sub_cols].mean(axis=1)
    elif method == "sum":
        result["rsvi"] = result[sub_cols].sum(axis=1) / len(sub_cols)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    logger.info(
        f"Computed composite RSVI from {len(sub_cols)} sub-indices "
        f"(method: {method}, "
        f"range: {result['rsvi'].min():.3f} – {result['rsvi'].max():.3f})"
    )
    return result


def build_rsvi(
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Full pipeline: load socioeconomic data → compute RSVI.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Panel data with RSVI and all sub-components.
    """
    if cfg is None:
        cfg = load_config()

    socio_path = get_path(cfg, "interim_data") / "socioeconomic_processed.parquet"
    output_path = get_path(cfg, "interim_data") / "rsvi.parquet"

    df = load_dataframe(socio_path)

    rsvi_cfg = cfg["rsvi"]
    sub_indices = rsvi_cfg["sub_indices"]

    # Collect all indicator columns
    all_indicators = []
    for indicators in sub_indices.values():
        all_indicators.extend(indicators)

    # Step 1: Percentile rank
    df = percentile_rank_within_year(df, all_indicators)

    # Step 2: Sub-indices
    df = compute_sub_indices(df, sub_indices)

    # Step 3: Composite RSVI
    sub_names = list(sub_indices.keys())
    df = compute_composite_rsvi(df, sub_names, method=rsvi_cfg["aggregation"])

    # Keep only the key columns for merging
    output_cols = ["nuts2_code", "year", "rsvi"]
    output_cols += [f"subidx_{name}" for name in sub_names if f"subidx_{name}" in df.columns]
    output_cols += all_indicators  # Keep raw indicators too

    available_cols = [c for c in output_cols if c in df.columns]
    output_df = df[available_cols].copy()

    save_dataframe(output_df, output_path, index=False)
    return output_df


if __name__ == "__main__":
    build_rsvi()
