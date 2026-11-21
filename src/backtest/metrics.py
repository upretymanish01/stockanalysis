"""Performance statistics for an equity curve.

Annualisation uses 252 trading days. Sharpe and Sortino assume a zero risk-free
rate. Sortino's downside deviation is the square root of the mean squared
negative return (zeros on up days stay in the average).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_metrics(
    equity: pd.Series,
    turnover: float,
    total_costs: float,
    n_rebalances: int,
) -> dict:
    """Summarise a marked-to-close equity curve."""
    curve = pd.Series(equity, dtype=float).dropna()
    if curve.empty:
        raise ValueError("Equity curve is empty")
    returns = curve.pct_change().dropna()
    start = curve.index[0]
    end = curve.index[-1]
    years = max(len(curve) - 1, 0) / TRADING_DAYS
    total_return = float(curve.iloc[-1] / curve.iloc[0] - 1.0) if curve.iloc[0] else float("nan")
    if years > 0 and curve.iloc[0] > 0 and curve.iloc[-1] > 0:
        cagr = float((curve.iloc[-1] / curve.iloc[0]) ** (1.0 / years) - 1.0)
    else:
        cagr = float("nan")
    if len(returns) >= 2 and float(returns.std(ddof=1)) > 0:
        ann_vol = float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
        sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
    else:
        ann_vol = float("nan")
        sharpe = float("nan")
    sortino = _sortino(returns)
    max_drawdown = _max_drawdown(curve)
    hit_rate = float((returns > 0).mean()) if len(returns) else float("nan")
    ann_turnover = float(turnover / years) if years > 0 else float("nan")
    return {
        "start": str(pd.Timestamp(start).date()),
        "end": str(pd.Timestamp(end).date()),
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "turnover": float(turnover),
        "ann_turnover": ann_turnover,
        "hit_rate": hit_rate,
        "total_costs": float(total_costs),
        "ending_equity": float(curve.iloc[-1]),
        "n_rebalances": int(n_rebalances),
    }


def _sortino(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
    deviation = float(np.sqrt(np.mean(downside**2)))
    if deviation == 0.0:
        if float(returns.mean()) > 0:
            return float("inf")
        return float("nan")
    return float(returns.mean() / deviation * math.sqrt(TRADING_DAYS))


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())
