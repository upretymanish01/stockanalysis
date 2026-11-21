"""Dead tickers are updated, and failed fetches are logged instead of ignored."""

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics import (
    FOCUS_CATEGORIES,
    SECTOR_ETF,
    _relative_strength,
    analytics_for_universe,
    get_tickers_for_focus,
    multi_period_universe,
)
from src.data_fetcher import fetch_price_history


def _history(n: int = 400) -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = np.linspace(100.0, 140.0, n)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, 2_000_000.0),
        },
        index=idx,
    )


class _Info:
    def __init__(self, ticker):
        self.ticker = ticker

    @property
    def info(self):
        return {"sector": "Financial Services", "industry": "Credit", "earningsQuarterlyGrowth": 0.05}


def test_finance_universe_uses_block_xyz_not_retired_sq():
    finance = get_tickers_for_focus("finance")
    assert finance == ["V", "MA", "PYPL", "XYZ", "AXP"]
    listed = {ticker for tickers in FOCUS_CATEGORIES.values() for ticker in tickers}
    assert "SQ" not in listed
    assert "XYZ" in listed


def test_education_benchmark_is_not_liquidated_edut():
    assert SECTOR_ETF["education"] == "XLY"
    assert "EDUT" not in SECTOR_ETF.values()


def test_empty_price_fetch_logs_a_warning(caplog):
    with patch("src.data_fetcher.yf.Ticker") as ticker_cls:
        ticker_cls.return_value.history.return_value = pd.DataFrame()
        with caplog.at_level(logging.WARNING):
            df = fetch_price_history("EDUT", period="2y")

    assert df.empty
    assert "EDUT" in caplog.text
    assert "No price data" in caplog.text


def test_price_fetch_exception_is_logged_and_reraised(caplog):
    with patch("src.data_fetcher.yf.Ticker") as ticker_cls:
        ticker_cls.return_value.history.side_effect = RuntimeError("possibly delisted")
        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match="possibly delisted"):
                fetch_price_history("SQ")

    assert "SQ" in caplog.text
    assert "Price fetch failed" in caplog.text


def test_universe_summarizes_skipped_tickers(caplog):
    good = _history()

    def fake_fetch(ticker, period="2y", interval="1d"):
        if ticker == "SQ":
            return pd.DataFrame()
        return good

    with patch("src.analytics.fetch_price_history", side_effect=fake_fetch), patch(
        "src.analytics.yf.Ticker", _Info
    ):
        with caplog.at_level(logging.WARNING):
            df = multi_period_universe(["AAPL", "SQ"], focus="finance")

    assert set(df.loc[df["ticker"] == "AAPL", "timeframe"]) == {
        "2d",
        "3d",
        "4d",
        "5d / 1w",
        "1m",
        "3m",
        "6m",
        "1y",
    }
    sq = df[df["ticker"] == "SQ"]
    assert sq["error"].notna().all()
    summary = [r.message for r in caplog.records if r.message.startswith("Skipped ")]
    assert summary == ["Skipped 1 ticker(s) due to data fetch failure: SQ"]
    assert any("No usable price history for SQ" in r.message for r in caplog.records)


def test_universe_logs_and_continues_when_a_fetch_raises(caplog):
    good = _history()

    def fake_fetch(ticker, period="2y", interval="1d"):
        if ticker == "SQ":
            raise RuntimeError("SQ delisted")
        return good

    with patch("src.analytics.fetch_price_history", side_effect=fake_fetch), patch(
        "src.analytics.yf.Ticker", _Info
    ):
        with caplog.at_level(logging.WARNING):
            df = multi_period_universe(["SQ", "V"], focus="finance")

    assert "V" in set(df["ticker"])
    sq = df[df["ticker"] == "SQ"].iloc[0]
    assert "SQ delisted" in str(sq["error"])
    assert "Skipped 1 ticker(s) due to data fetch failure: SQ" in caplog.text
    assert "SQ delisted" in caplog.text


def test_relative_strength_logs_a_missing_benchmark(caplog):
    good = _history()

    def fake_fetch(ticker, period="1y", interval="1d"):
        if ticker == "EDUT":
            return pd.DataFrame()
        return good

    with patch("src.analytics.fetch_price_history", side_effect=fake_fetch):
        with caplog.at_level(logging.WARNING):
            value = _relative_strength("DUOL", "EDUT", period="1y")

    assert np.isnan(value)
    assert "EDUT" in caplog.text
    assert "Relative strength unavailable" in caplog.text


def test_analytics_universe_summarizes_tickers_with_no_data(caplog):
    good = _history()

    def fake_fetch(ticker, period="1y", interval="1d"):
        if ticker == "EDUT":
            return pd.DataFrame()
        return good

    with patch("src.analytics.fetch_price_history", side_effect=fake_fetch), patch(
        "src.quality_screen.yf.Ticker", _Info
    ):
        with caplog.at_level(logging.WARNING):
            df = analytics_for_universe(["AAPL", "EDUT"], period="1y")

    assert "EDUT" in set(df["ticker"])
    assert df.loc[df["ticker"] == "EDUT", "error"].notna().all()
    assert "AAPL" in set(df["ticker"])
    assert "Skipped 1 ticker(s) due to data fetch failure: EDUT" in caplog.text
