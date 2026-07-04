"""
risk_metrics.py
Value at Risk (VaR) and Sharpe Ratio calculations.
"""

import numpy as np
import pandas as pd


def value_at_risk(returns: pd.Series, confidence=0.95) -> float:
    """
    Historical Value at Risk: the loss threshold that returns fall below
    only (1 - confidence) of the time. E.g. at 95% confidence, this is the
    5th percentile of the return distribution.
    """
    returns = returns.dropna()
    percentile = (1 - confidence) * 100
    return np.percentile(returns, percentile)


def sharpe_ratio(returns: pd.Series, risk_free_rate=0.02, periods_per_year=252) -> float:
    """
    Annualized Sharpe Ratio: excess return per unit of risk.
    """
    returns = returns.dropna()
    daily_rf = risk_free_rate / periods_per_year
    excess_return = returns.mean() - daily_rf
    return (excess_return / returns.std()) * np.sqrt(periods_per_year)
