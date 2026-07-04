"""
data_loader.py
Fetches historical price data for TSLA, BND, and SPY from Yahoo Finance
and saves each as a CSV file under data/processed/.
"""

import os
import time
import yfinance as yf
import pandas as pd

TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-06-30"
OUTPUT_DIR = "data/processed"
REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def fetch_ticker_data(ticker: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame:
    """
    Fetch historical data for a single ticker from Yahoo Finance, retrying
    on transient failures (e.g. network errors, rate limits).

    Raises:
        RuntimeError: if all retry attempts fail.
        ValueError: if the returned data is empty or missing required columns.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end)

            if df.empty:
                raise ValueError(f"No data returned for {ticker} (empty result)")

            # Flatten multi-index columns (newer yfinance versions return
            # (field, ticker) tuples even for a single ticker).
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            missing_cols = REQUIRED_COLUMNS - set(df.columns)
            if missing_cols:
                raise ValueError(f"{ticker} data is missing expected columns: {missing_cols}")

            df.index.name = "Date"
            return df

        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}/{max_retries} failed for {ticker}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)  # simple exponential backoff

    raise RuntimeError(f"Failed to fetch data for {ticker} after {max_retries} attempts: {last_error}")


def fetch_and_save_data():
    """Fetch and save data for all TICKERS. Continues past individual failures."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    failed_tickers = []

    for ticker in TICKERS:
        print(f"Fetching {ticker}...")
        try:
            df = fetch_ticker_data(ticker, START_DATE, END_DATE)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            failed_tickers.append(ticker)
            continue

        output_path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
        df.to_csv(output_path)
        print(f"  Saved {len(df)} rows to {output_path}")

    if failed_tickers:
        print(f"\nWARNING: Failed to fetch data for: {', '.join(failed_tickers)}")
    else:
        print("\nAll tickers fetched successfully.")


if __name__ == "__main__":
    fetch_and_save_data()
