"""Position size is based on each timeframe's VaR, including horizons other than 1m and 3m."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics import _suggested_position_pct, multi_period_returns


def _history(n: int = 400, mu: float = 0.08, sigma: float = 0.35, seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    close = 50.0 * np.exp(np.cumsum(shocks))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 1_500_000.0),
        },
        index=idx,
    )


class _Ticker:
    def __init__(self, ticker):
        self.ticker = ticker

    @property
    def info(self):
        return {
            "sector": "Technology",
            "industry": "Software",
            "earningsQuarterlyGrowth": 0.12,
        }


def test_missing_var_does_not_collapse_position_to_zero():
    # NaN is truthy, so `var or 0.05` used to keep NaN and `nan > 0` yielded 0%.
    assert _suggested_position_pct(np.nan) == pytest.approx(0.10)
    assert _suggested_position_pct(None) == pytest.approx(0.10)
    assert _suggested_position_pct(float("inf")) == pytest.approx(0.10)
    # A real 25% downside VaR risks 1% of capital -> 4% of capital, under the 10% cap.
    assert _suggested_position_pct(-0.25) == pytest.approx(0.04)
    assert _suggested_position_pct(-0.25) > 0


def test_position_size_is_nonzero_for_every_timeframe():
    history = _history()

    with patch("src.analytics.fetch_price_history", return_value=history), patch(
        "src.analytics.yf.Ticker", _Ticker
    ):
        result = multi_period_returns("AAPL", focus="tech")

    expected = ["2d", "3d", "4d", "5d / 1w", "1m", "3m", "6m", "1y"]
    assert result["timeframe"].tolist() == expected
    assert np.isfinite(result["mc_var_95"]).all()
    assert (result["suggested_position_pct"] > 0).all()
    assert (result["suggested_position_pct"] <= 10).all()
    # Horizon risk is not a single constant pasted onto every row.
    assert result["suggested_position_pct"].nunique() > 1
    outlook = set(result.loc[result["model_name"].notna(), "timeframe"])
    assert outlook == {"1m", "3m"}
