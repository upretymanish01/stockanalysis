"""Data fetching utilities for stock analysis."""

import logging

import yfinance as yf
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


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
        DataFrame with OHLCV data. Empty when the provider returns nothing.
        Failures are logged; exceptions are logged and re-raised.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
    except Exception as exc:
        logger.warning(
            "Price fetch failed for %s (period=%s, interval=%s): %s",
            ticker,
            period,
            interval,
            exc,
        )
        raise

    if df is None or df.empty:
        logger.warning(
            "No price data for %s (period=%s, interval=%s)",
            ticker,
            period,
            interval,
        )
        return pd.DataFrame() if df is None else df
    return df


def fetch_info(ticker: str) -> dict:
    """Fetch basic company info."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
    except Exception as exc:
        logger.warning("Info fetch failed for %s: %s", ticker, exc)
        raise
    if not info:
        logger.warning("No company info for %s", ticker)
        return {}
    return info
