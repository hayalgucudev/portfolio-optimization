"""Unit tests for src/data_cleaning.py"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest
from src.data_cleaning import (
    check_data_types,
    check_missing_values,
    add_daily_return,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [105.0, 106.0, 107.0],
            "Low": [95.0, 96.0, 97.0],
            "Close": [102.0, 103.0, 104.0],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )


def test_check_data_types(sample_df):
    dtypes = check_data_types(sample_df)
    assert dtypes["Close"] == "float64"
    assert dtypes["Volume"] == "int64"


def test_check_missing_values_none(sample_df):
    missing = check_missing_values(sample_df)
    assert missing.empty


def test_check_missing_values_detected(sample_df):
    sample_df.loc[sample_df.index[1], "Close"] = None
    missing = check_missing_values(sample_df)
    assert missing["Close"] == 1


def test_add_daily_return(sample_df):
    result = add_daily_return(sample_df)
    assert "Daily Return" in result.columns
    # First row has no prior day, so return should be NaN
    assert pd.isna(result["Daily Return"].iloc[0])
    # Second day: (103 - 102) / 102
    expected = (103.0 - 102.0) / 102.0
    assert abs(result["Daily Return"].iloc[1] - expected) < 1e-9
