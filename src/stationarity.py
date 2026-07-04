"""
stationarity.py
Augmented Dickey-Fuller stationarity testing.
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series, significance=0.05) -> dict:
    """
    Run the Augmented Dickey-Fuller test on a series.

    Returns a dict with the test statistic, p-value, critical values,
    and a plain-language conclusion.
    """
    series = series.dropna()
    result = adfuller(series)

    return {
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "critical_values": result[4],
        "is_stationary": bool(result[1] < significance),
    }
