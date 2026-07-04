# Portfolio Optimization — Time Series Forecasting for GMF Investments

Week 9 Challenge (10 Academy: Artificial Intelligence Mastery)

## Business Context

GMF Investments is a financial advisory firm exploring how time series
forecasting can support portfolio management. This project uses historical
data for three assets — **TSLA** (high-growth stock), **BND** (bond ETF,
low risk), and **SPY** (S&P 500 ETF, moderate risk) — to build forecasting
models and apply Modern Portfolio Theory to recommend an optimized
portfolio allocation.

## Project Structure
portfolio-optimization/
├── .vscode/                 # Editor settings
├── .github/workflows/       # CI configuration (unit tests)
├── data/
│   └── processed/           # Cleaned CSVs and saved plots (gitignored raw data)
├── notebooks/
│   ├── 1.0-eda.ipynb        # Task 1: data cleaning, EDA, stationarity, risk metrics
│   └── 2.0-arima-model.ipynb # Task 2: ARIMA baseline model
├── src/
│   └── data_loader.py       # Fetches TSLA/BND/SPY data from yfinance
├── tests/                   # Unit tests
├── scripts/                 # Utility/automation scripts
└── requirements.txt

## Setup

1. Clone the repository:
```bash
   git clone https://github.com/hayalgucudev/portfolio-optimization.git
   cd portfolio-optimization
```

2. Create and activate a virtual environment (**Python 3.11 required** —
   TensorFlow does not yet support 3.14+):
```bash
   py -3.11 -m venv venv
   source venv/Scripts/activate   # Windows Git Bash
   # source venv/bin/activate     # Mac/Linux
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Fetch the data:
```bash
   python src/data_loader.py
```

5. Launch Jupyter to explore the notebooks:
```bash
   jupyter notebook
```

## Data

Historical daily price data (Open, High, Low, Close, Volume) for TSLA, BND,
and SPY, sourced from Yahoo Finance via the `yfinance` library, covering
**January 1, 2015 – June 30, 2026**.

## Progress

### ✅ Task 1 — Preprocess and Explore the Data
- Extracted historical data for all three assets and saved to `data/processed/`
- Verified data types, confirmed zero missing values
- Performed EDA: closing price trends, daily returns, rolling volatility,
  outlier detection
- Ran Augmented Dickey-Fuller stationarity tests on price and return series
- Calculated Value at Risk (95%) and Sharpe Ratio for each asset

### 🔄 Task 2 — Build Time Series Forecasting Models (in progress)
- Split TSLA data chronologically (train: 2015–2024, test: 2025–2026)
- Fit an ARIMA(0,1,0) baseline model via `auto_arima`
- Evaluated with MAE ($54.15), RMSE ($70.20), MAPE (17.11%)
- **Next:** implement and tune an LSTM model for comparison

### ⬜ Task 3 — Forecast Future Market Trends
### ⬜ Task 4 — Optimize Portfolio Based on Forecast
### ⬜ Task 5 — Strategy Backtesting

## Key Findings So Far

- All three assets' closing prices are non-stationary; daily returns are
  stationary — confirming `d=1` is appropriate for ARIMA modeling.
- The best-fit ARIMA model for TSLA is (0,1,0), a random-walk model with no
  autoregressive or moving-average signal — an empirical illustration of
  the Efficient Market Hypothesis referenced in the project brief.
- TSLA carries the highest risk (VaR: -5.11%) but also the highest
  risk-adjusted return (Sharpe: 0.74) of the three assets over this period.

## Team

Kerod · Mahbubah · Feven

## References

See the full challenge document for tutorials schedule and external reading
links on ARIMA, LSTM, Modern Portfolio Theory, and backtesting.