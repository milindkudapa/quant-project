"""
Process mortality data specifically from the massive ISTAT municipal daily datasets.

Handles loading, cleaning, and aggregating mortality data to NUTS-2 regional
level for use in the panel analysis, using memory-efficient wide-to-long
melting.

Usage
-----
    python -m src.data.process_istat
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.utils.config import load_config, get_path
from src.utils.constants import SUMMER_MONTHS
from src.utils.io import save_dataframe


def load_istat_mortality(filepath: Path) -> pd.DataFrame:
    """Load ISTAT daily mortality data.

    Processes 5 million+ row wide-format municipal CSV files optimally by
    selecting required columns, pre-filtering on summer months via GE string,
    melting to long format, and directly outputting region-level aggregates.
    """
    logger.info(f"Loading ISTAT wide-format mortality data from {filepath}")

    # The dataset contains years as T_11...T_24. We care about 2012-2022.
    years = list(range(12, 23))
    t_cols = [f"T_{y}" for y in years]
    use_cols = ["NOME_REGIONE", "GE"] + t_cols
    
    # Read the data, ensuring GE is read as string
    df = pd.read_csv(
        filepath, encoding="latin1", usecols=use_cols, 
        dtype={"GE": str}, na_values="n.d."
    )
    
    # Filter for summer months: GE starts with 06, 07, 08, 09
    df = df[df["GE"].str.match(r'^(06|07|08|09)')].copy()
    
    # Map NOME_REGIONE to nuts2_code using Eurostat NUTS-2 2021 codes.
    # Trentino-Alto Adige is reported as one region by ISTAT; it maps to ITH1
    # (which in the climate data is the merged ITH1+ITH2 polygon).
    nuts_map = {
        "Piemonte": "ITC1",
        "Valle d'Aosta/Vallée d'Aoste": "ITC2",
        "Liguria": "ITC3",
        "Lombardia": "ITC4",
        "Trentino-Alto Adige/Südtirol": "ITH1",  # merged Bolzano+Trento
        "Veneto": "ITH3",
        "Friuli-Venezia Giulia": "ITH4",
        "Emilia-Romagna": "ITH5",
        "Toscana": "ITI1",
        "Umbria": "ITI2",
        "Marche": "ITI3",
        "Lazio": "ITI4",
        "Abruzzo": "ITF1",
        "Molise": "ITF2",
        "Campania": "ITF3",
        "Puglia": "ITF4",
        "Basilicata": "ITF5",
        "Calabria": "ITF6",
        "Sicilia": "ITG1",
        "Sardegna": "ITG2"
    }
    df["nuts2_code"] = df["NOME_REGIONE"].map(nuts_map)
    
    # Melt from wide to long
    melted = df.melt(
        id_vars=["nuts2_code", "GE"], 
        value_vars=t_cols, 
        var_name="year_col", 
        value_name="deaths"
    )
    
    # Extract temporal data
    melted["year"] = 2000 + melted["year_col"].str.split("_").str[1].astype(int)
    melted["month"] = melted["GE"].str[:2].astype(int)
    
    # Sum deaths
    melted["deaths"] = pd.to_numeric(melted["deaths"], errors="coerce")
    monthly = melted.groupby(["nuts2_code", "year", "month"], as_index=False)["deaths"].sum()
    
    # Create proxy date to satisfy downstream pipeline signature requirements
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2) + "-01"
    )
    
    logger.info(f"Loaded {len(monthly)} ISTAT mortality region-month records")
    return monthly


def compute_summer_mortality(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compute summer (June–September) total mortality per region-year.

    Parameters
    ----------
    monthly : pd.DataFrame
        Monthly mortality with columns: nuts2_code, year, month, deaths.

    Returns
    -------
    pd.DataFrame
        Summer mortality per region-year.
    """
    summer = monthly[monthly["month"].isin(SUMMER_MONTHS)].copy()

    # Aggregate across summer months (and optionally across age/sex for totals)
    group_cols = ["nuts2_code", "year"]
    summer_totals = (
        summer.groupby(group_cols, as_index=False)["deaths"]
        .sum()
        .rename(columns={"deaths": "summer_deaths"})
    )

    logger.info(f"Computed summer mortality for {len(summer_totals)} region-years")
    return summer_totals


def process_istat_data(
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Full pipeline: load and process ISTAT municipal mortality data.

    Parameters
    ----------
    cfg : dict, optional
        Configuration dictionary.

    Returns
    -------
    pd.DataFrame
        Processed mortality data ready for panel assembly.
    """
    if cfg is None:
        cfg = load_config()

    raw_dir = get_path(cfg, "raw_data") / "mortality"
    output_path = get_path(cfg, "interim_data") / "mortality_processed.parquet"

    raw_files = list(raw_dir.rglob("*.csv")) + list(raw_dir.rglob("*.xlsx"))

    istat_file = next((f for f in raw_files if "comuni_giornaliero" in f.name.lower()), None)
    
    if istat_file:
        logger.info(f"Found ISTAT municipal dataset at {istat_file}. Processing...")
        combined = load_istat_mortality(istat_file)
    else:
        logger.error("No ISTAT 'comuni_giornaliero' file found. Run 'process_mortality' to use Eurostat instead.")
        return pd.DataFrame()

    summer = compute_summer_mortality(combined)
    save_dataframe(summer, output_path, index=False)
    save_dataframe(
        summer, 
        get_path(cfg, "interim_data") / "mortality_processed_istat.parquet", 
        index=False
    )
    
    return summer


if __name__ == "__main__":
    process_istat_data()
