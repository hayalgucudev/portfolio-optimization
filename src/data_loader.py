"""
data_loader.py
Fetches historical price data for TSLA, BND, and SPY from Yahoo Finance
and saves each as a CSV file under data/processed/.
"""

import yfinance as yf
import pandas as pd
import os

TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-06-30"
OUTPUT_DIR = "data/processed"


def fetch_and_save_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ticker in TICKERS:
        print(f"Fetching {ticker}...")
        df = yf.download(ticker, start=START_DATE, end=END_DATE)

        if df.empty:
            print(f"  WARNING: No data returned for {ticker}")
            continue

        # Newer yfinance versions return multi-index columns like
        # ('Close', 'TSLA') even for a single ticker. Flatten them
        # down to just 'Close', 'Open', etc. so the CSV is simple.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "Date"  # make sure the index column is explicitly named

        output_path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
        df.to_csv(output_path)
        print(f"  Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    fetch_and_save_data()
