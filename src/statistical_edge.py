"""Lightweight statistical tools for testing simple edges or signals.

Inspired by systematic approaches (Simons-style process, not the models).
Useful for checking whether a rule has historical edge before relying on it.
"""

from typing import Optional

import numpy as np
import pandas as pd


def compute_returns(prices: pd.Series) -> pd.Series:
    """Simple daily returns."""
    return prices.pct_change().dropna()


def signal_hit_rate(
    prices: pd.Series,
    signal: pd.Series,
    forward_days: int = 5,
) -> dict:
    """
    Measure how often a binary signal is followed by positive forward returns.

    Args:
        prices: Price series (aligned index with signal)
        signal: Boolean or 0/1 series (True / 1 = long signal)
        forward_days: Horizon to measure forward return

    Returns:
        Dictionary with hit rate, average forward return, count, etc.
    """
    returns = compute_returns(prices)
    # Align
    df = pd.DataFrame({"ret": returns, "signal": signal}).dropna()
    if df.empty:
        return {"error": "No overlapping data"}

    # Forward return
    df["fwd_ret"] = df["ret"].shift(-forward_days)
    df = df.dropna()

    signaled = df[df["signal"] == True]  # or == 1
    if len(signaled) == 0:
        return {"error": "No signals triggered"}

    hit_rate = (signaled["fwd_ret"] > 0).mean()
    avg_fwd = signaled["fwd_ret"].mean()
    baseline_avg = df["fwd_ret"].mean()

    return {
        "n_signals": int(len(signaled)),
        "hit_rate": float(hit_rate),
        "avg_forward_return": float(avg_fwd),
        "baseline_avg_return": float(baseline_avg),
        "edge_vs_baseline": float(avg_fwd - baseline_avg),
        "forward_days": forward_days,
    }


def simple_momentum_signal(
    prices: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Example binary signal: price above its N-day moving average."""
    ma = prices.rolling(lookback).mean()
    return (prices > ma).astype(int)


def test_momentum_edge(
    prices: pd.Series,
    lookback: int = 20,
    forward_days: int = 5,
) -> dict:
    """Quick test of a simple momentum rule."""
    signal = simple_momentum_signal(prices, lookback)
    return signal_hit_rate(prices, signal, forward_days)
