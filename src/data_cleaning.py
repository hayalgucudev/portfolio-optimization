"""
data_cleaning.py
Functions for validating and cleaning raw price data loaded from CSV.
"""

import pandas as pd

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def load_asset_csv(filepath: str) -> pd.DataFrame:
    """
    Load a single asset's CSV file, parsing the Date column as the index.

    Raises:
        FileNotFoundError: if the file doesn't exist.
        ValueError: if required columns are missing.
    """
    try:
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data file: {filepath}")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"{filepath} is missing required columns: {missing_cols}")

    return df


def check_data_types(df: pd.DataFrame) -> pd.Series:
    """Return the dtype of each column, for quick inspection/logging."""
    return df.dtypes


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return count of missing values per column (only columns with >0 shown)."""
    missing = df.isnull().sum()
    return missing[missing > 0]


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex to business-day frequency and forward-fill any resulting gaps.
    Financial data typically has no missing values within trading days, but
    this guards against holidays/gaps introduced by the reindex.
    """
    df = df.asfreq("B")
    df = df.ffill()
    return df


def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'Daily Return' column: day-over-day percentage change in Close."""
    df = df.copy()
    df["Daily Return"] = df["Close"].pct_change()
    return df
