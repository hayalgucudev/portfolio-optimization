"""Unit tests for src/risk_metrics.py"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from src.risk_metrics import value_at_risk, sharpe_ratio


def test_value_at_risk_known_distribution():
    # Returns from -10% to +10% in 1% steps: 5th percentile should be near -9%
    returns = pd.Series(np.arange(-0.10, 0.11, 0.01))
    var = value_at_risk(returns, confidence=0.95)
    assert var < 0  # VaR should represent a loss
    assert -0.10 <= var <= -0.05


def test_sharpe_ratio_positive_returns():
    # Consistent small positive daily returns should give a positive Sharpe Ratio
    returns = pd.Series([0.001] * 100)
    ratio = sharpe_ratio(returns, risk_free_rate=0.0)
    assert ratio > 0


def test_sharpe_ratio_near_zero_when_returns_match_risk_free():
    # Returns fluctuating randomly around the risk-free rate should give
    # a Sharpe Ratio close to zero (small in magnitude either way).
    np.random.seed(42)
    daily_rf = 0.02 / 252
    returns = pd.Series(np.random.normal(loc=daily_rf, scale=0.001, size=252))
    ratio = sharpe_ratio(returns, risk_free_rate=0.02)
    assert abs(ratio) < 1.0
