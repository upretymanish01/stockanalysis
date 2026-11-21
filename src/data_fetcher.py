"""Data fetching utilities for stock analysis."""

import logging

import yfinance as yf
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def fetch_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch historical price data for a ticker.

    Args:
        ticker: Stock symbol (e.g. 'AAPL')
        period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max.
            Ignored when start or end is set.
        interval: Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        start: Inclusive range start (YYYY-MM-DD). Uses the provider's range
            query instead of period.
        end: Exclusive range end, matching yfinance. Pass the day after the
            last session you want.

    Returns:
        DataFrame with OHLCV data. Empty when the provider returns nothing.
        Failures are logged; exceptions are logged and re-raised.
    """
    kwargs = {"interval": interval}
    if start is not None or end is not None:
        if start is not None:
            kwargs["start"] = start
        if end is not None:
            kwargs["end"] = end
    else:
        kwargs["period"] = period
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(**kwargs)
    except Exception as exc:
        logger.warning(
            "Price fetch failed for %s (period=%s, interval=%s, start=%s, end=%s): %s",
            ticker,
            period,
            interval,
            start,
            end,
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
