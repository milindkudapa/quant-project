"""Data I/O helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


def save_dataframe(df: pd.DataFrame, path: Path | str, **kwargs) -> None:
    """Save a DataFrame, auto-detecting format from the file extension.

    Supported extensions: ``.csv``, ``.parquet``, ``.xlsx``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : Path or str
        Destination file path.
    **kwargs
        Additional keyword arguments passed to the pandas writer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=kwargs.pop("index", True), **kwargs)
    elif ext == ".parquet":
        df.to_parquet(path, index=kwargs.pop("index", True), **kwargs)
    elif ext == ".xlsx":
        df.to_excel(path, index=kwargs.pop("index", True), **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    logger.info(f"Saved {len(df)} rows → {path}")


def load_dataframe(path: Path | str, **kwargs) -> pd.DataFrame:
    """Load a DataFrame, auto-detecting format from the file extension.

    Parameters
    ----------
    path : Path or str
        Source file path.
    **kwargs
        Additional keyword arguments passed to the pandas reader.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path, **kwargs)
    elif ext == ".parquet":
        df = pd.read_parquet(path, **kwargs)
    elif ext == ".xlsx":
        df = pd.read_excel(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    logger.info(f"Loaded {len(df)} rows ← {path}")
    return df


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (and parents) if it doesn't exist.

    Parameters
    ----------
    path : Path or str
        Directory to create.

    Returns
    -------
    Path
        The created/existing directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
