"""
eda.py
Exploratory data analysis functions: summary stats, rolling volatility,
and outlier detection on returns.
"""

import pandas as pd


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics (count, mean, std, min, max, quartiles)."""
    return df.describe()


def rolling_mean_std(df: pd.DataFrame, price_col="Close", return_col="Daily Return", window=30):
    """
    Compute rolling mean of price and rolling std of returns over `window` days.
    Returns a tuple: (rolling_mean_series, rolling_std_series).
    """
    rolling_mean = df[price_col].rolling(window=window).mean()
    rolling_std = df[return_col].rolling(window=window).std()
    return rolling_mean, rolling_std


def detect_outliers(df: pd.DataFrame, return_col="Daily Return", threshold=3.0) -> pd.Series:
    """
    Flag days where the return is more than `threshold` standard deviations
    from the mean. Returns a Series of just the outlier values, indexed by date.
    """
    returns = df[return_col].dropna()
    mean, std = returns.mean(), returns.std()
    outliers = returns[(returns - mean).abs() > threshold * std]
    return outliers.sort_values()
