"""
Temperature feature engineering.

Computes derived temperature variables including apparent temperature
(heat index), wet-bulb globe temperature approximation, and seasonal
anomalies for use in the panel analysis.

Usage
-----
    python -m src.features.temperature
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_relative_humidity(
    t2m: np.ndarray | pd.Series,
    dewpoint: np.ndarray | pd.Series,
) -> np.ndarray | pd.Series:
    """Compute relative humidity from temperature and dewpoint.

    Uses the Magnus formula approximation.

    Parameters
    ----------
    t2m : array-like
        Air temperature in °C.
    dewpoint : array-like
        Dewpoint temperature in °C.

    Returns
    -------
    array-like
        Relative humidity as a percentage (0–100).
    """
    b = 17.625
    c = 243.04  # °C

    gamma_t = (b * t2m) / (c + t2m)
    gamma_d = (b * dewpoint) / (c + dewpoint)

    rh = 100.0 * np.exp(gamma_d - gamma_t)
    return np.clip(rh, 0, 100)


def compute_apparent_temperature(
    t2m: np.ndarray | pd.Series,
    rh: np.ndarray | pd.Series,
) -> np.ndarray | pd.Series:
    """Compute apparent temperature (heat index).

    Uses the Steadman (1979) / NWS regression equation. This is the
    perceived temperature accounting for humidity effects on the body.

    Parameters
    ----------
    t2m : array-like
        Air temperature in °C.
    rh : array-like
        Relative humidity as a percentage.

    Returns
    -------
    array-like
        Apparent temperature in °C.
    """
    # Convert to Fahrenheit for the NWS formula
    tf = t2m * 9 / 5 + 32

    # Simple heat index (Rothfusz regression)
    hi = (
        -42.379
        + 2.04901523 * tf
        + 10.14333127 * rh
        - 0.22475541 * tf * rh
        - 0.00683783 * tf**2
        - 0.05481717 * rh**2
        + 0.00122874 * tf**2 * rh
        + 0.00085282 * tf * rh**2
        - 0.00000199 * tf**2 * rh**2
    )

    # Convert back to Celsius
    hi_c = (hi - 32) * 5 / 9

    # For low temperatures, just use the air temperature
    mask = t2m < 27  # Heat index is only valid above ~27°C / 80°F
    if isinstance(hi_c, pd.Series):
        hi_c = hi_c.where(~mask, t2m)
    else:
        hi_c = np.where(mask, t2m, hi_c)

    return hi_c


def compute_wbgt_approximation(
    t2m: np.ndarray | pd.Series,
    dewpoint: np.ndarray | pd.Series,
) -> np.ndarray | pd.Series:
    """Approximate Wet-Bulb Globe Temperature (WBGT).

    Uses the simplified Liljegren et al. approximation for outdoor WBGT,
    which requires only temperature and humidity (no wind or radiation).

    WBGT ≈ 0.567 × T + 0.393 × e + 3.94

    where e is the water vapor pressure in hPa.

    Parameters
    ----------
    t2m : array-like
        Air temperature in °C.
    dewpoint : array-like
        Dewpoint temperature in °C.

    Returns
    -------
    array-like
        Approximate WBGT in °C.
    """
    # Water vapor pressure (hPa) from dewpoint using Buck equation
    e = 6.112 * np.exp((17.67 * dewpoint) / (dewpoint + 243.5))

    wbgt = 0.567 * t2m + 0.393 * e + 3.94
    return wbgt


def add_temperature_features(
    daily_climate: pd.DataFrame,
) -> pd.DataFrame:
    """Add derived temperature features to the daily climate DataFrame.

    Adds: relative humidity, apparent temperature, WBGT approximation.

    Parameters
    ----------
    daily_climate : pd.DataFrame
        Daily climate data with tmax, tmean, dewpoint_mean columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with additional temperature feature columns.
    """
    df = daily_climate.copy()

    # Relative humidity (based on daily mean values)
    df["rh_mean"] = compute_relative_humidity(df["tmean"], df["dewpoint_mean"])

    # Apparent temperature (based on daily max + mean humidity)
    df["apparent_tmax"] = compute_apparent_temperature(df["tmax"], df["rh_mean"])

    # WBGT approximation (based on daily mean)
    df["wbgt_mean"] = compute_wbgt_approximation(df["tmean"], df["dewpoint_mean"])
    df["wbgt_max"] = compute_wbgt_approximation(df["tmax"], df["dewpoint_mean"])

    logger.info(
        f"Added temperature features: rh_mean, apparent_tmax, wbgt_mean, wbgt_max "
        f"({len(df)} rows)"
    )
    return df
