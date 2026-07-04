"""Unit tests for src/stationarity.py"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from src.stationarity import adf_test


def test_adf_detects_stationary_series():
    # Pure random noise around zero should be stationary
    np.random.seed(42)
    stationary_series = pd.Series(np.random.normal(0, 1, 500))
    result = adf_test(stationary_series)
    assert result["is_stationary"] == True


def test_adf_detects_nonstationary_series():
    # A random walk (cumulative sum of noise) should be non-stationary
    np.random.seed(42)
    random_walk = pd.Series(np.random.normal(0, 1, 500)).cumsum()
    result = adf_test(random_walk)
    assert result["is_stationary"] == False
