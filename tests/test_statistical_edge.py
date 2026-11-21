"""The edge tester measures the cumulative N-day forward return."""

import numpy as np
import pandas as pd
import pytest

from src.statistical_edge import signal_hit_rate, simple_momentum_signal
from src.statistical_edge import test_momentum_edge as run_momentum_edge


def test_forward_return_is_cumulative_not_the_single_session():
    # From t=1, two sessions ahead lands on 121.
    # Cumulative: 121/130 - 1 = -6.9% (a loss).
    # The old shift(-2) on daily returns was the day-3 session only: 121/110 - 1 = +10%.
    prices = pd.Series([100.0, 130.0, 110.0, 121.0])
    signal = pd.Series([0, 1, 0, 0])

    result = signal_hit_rate(prices, signal, forward_days=2)

    cumulative = 121.0 / 130.0 - 1.0
    single_session = 121.0 / 110.0 - 1.0
    assert result["n_signals"] == 1
    assert result["forward_days"] == 2
    assert result["avg_forward_return"] == pytest.approx(cumulative)
    assert result["hit_rate"] == 0.0
    assert cumulative < 0 < single_session
    assert result["avg_forward_return"] != pytest.approx(single_session)


def test_average_of_several_signals_matches_holding_period_returns():
    prices = pd.Series([100.0, 110.0, 90.0, 99.0, 120.0])
    signal = pd.Series([1, 1, 0, 0, 0])
    forward_days = 2

    result = signal_hit_rate(prices, signal, forward_days=forward_days)

    fwd = prices.shift(-forward_days) / prices - 1
    signaled = fwd.iloc[:2]
    assert result["avg_forward_return"] == pytest.approx(signaled.mean())
    assert result["avg_forward_return"] == pytest.approx(-0.10)
    assert result["n_signals"] == 2
    # Baseline is every bar that still has a full window, not only the signals.
    assert result["baseline_avg_return"] == pytest.approx(fwd.dropna().mean())


def test_one_day_horizon_is_the_next_session_return():
    prices = pd.Series([100.0, 80.0, 120.0])
    result = signal_hit_rate(prices, pd.Series([1, 0, 0]), forward_days=1)
    assert result["avg_forward_return"] == pytest.approx(80.0 / 100.0 - 1.0)
    assert result["hit_rate"] == 0.0


def test_momentum_edge_uses_cumulative_forward_return():
    prices = pd.Series(np.linspace(100.0, 180.0, 80))
    forward_days = 5
    lookback = 20

    result = run_momentum_edge(prices, lookback=lookback, forward_days=forward_days)

    signal = simple_momentum_signal(prices, lookback)
    fwd = prices.shift(-forward_days) / prices - 1
    expected = fwd[signal.astype(bool)].dropna()
    assert result["n_signals"] == len(expected)
    assert result["avg_forward_return"] == pytest.approx(float(expected.mean()))
    # A 5-day move on this ramp is several times one session, so a single-day
    # definition cannot match the cumulative holding-period return.
    one_session = prices.pct_change().dropna().mean()
    assert result["avg_forward_return"] > one_session * 2


def test_forward_days_must_be_positive():
    result = signal_hit_rate(pd.Series([1.0, 2.0, 3.0]), pd.Series([1, 1, 1]), forward_days=0)
    assert "error" in result
