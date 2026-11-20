"""Data fetching utilities for stock analysis."""

import yfinance as yf
import pandas as pd
from typing import Optional


def fetch_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Fetch historical price data for a ticker.

    Args:
        ticker: Stock symbol (e.g. 'AAPL')
        period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

    Returns:
        DataFrame with OHLCV data
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df


def fetch_info(ticker: str) -> dict:
    """Fetch basic company info."""
    stock = yf.Ticker(ticker)
    return stock.info
