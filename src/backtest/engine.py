"""Event-driven daily backtest.

A strategy sees prices only through the decision close. The resulting target
weights are filled at the next session's open. Commission is cash; slippage
moves the fill against the trade, both in basis points of traded notional.

Missing prints do not drop a holding out of the valuation. The last positive
price is used, and the day is logged as a stale mark. After ``delist_after``
consecutive sessions with neither an open nor a close, the shares are written
off at zero and the loss is logged. A name that cannot be traded is not treated
as cash that can be spent on something else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .panel import PricePanel


class Strategy(Protocol):
    name: str
    warmup: int

    def target_weights(self, as_of: pd.Timestamp, history: PricePanel) -> pd.Series:
        """Long-only weights for the close of ``as_of``. Need not sum to 1."""


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    rebalance: str = "monthly"
    trade_start: Optional[pd.Timestamp] = None
    # Consecutive sessions with no open and no close before a held name is
    # written off at zero. Suspensions shorter than this stay on the last print.
    delist_after: int = 10

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("commission_bps and slippage_bps must be non-negative")
        if self.rebalance not in {"daily", "weekly", "monthly"}:
            raise ValueError("rebalance must be daily, weekly, or monthly")
        if self.delist_after < 0:
            raise ValueError("delist_after must be >= 0 (0 disables write-offs)")
        if self.trade_start is not None:
            self.trade_start = pd.Timestamp(self.trade_start).normalize()


@dataclass
class BacktestResult:
    name: str
    sample: str
    equity: pd.Series
    returns: pd.Series
    metrics: dict
    total_costs: float
    turnover: float
    decision_log: list[dict] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    valuation_events: list[dict] = field(default_factory=list)


def decision_dates(
    index: pd.DatetimeIndex,
    freq: str,
    warmup: int,
    trade_start: Optional[pd.Timestamp] = None,
) -> pd.DatetimeIndex:
    """Rebalance closes. The last bar is excluded because there is no next open."""
    if freq not in {"daily", "weekly", "monthly"}:
        raise ValueError("rebalance must be daily, weekly, or monthly")
    if len(index) < 2:
        return pd.DatetimeIndex([])
    warmup = max(int(warmup), 0)
    eligible = index[min(warmup, len(index)) :]
    if trade_start is not None:
        eligible = eligible[eligible >= pd.Timestamp(trade_start)]
    eligible = eligible[eligible < index[-1]]
    if len(eligible) == 0:
        return pd.DatetimeIndex([])
    if freq == "daily":
        chosen = eligible
    elif freq == "weekly":
        iso = eligible.isocalendar()
        keys = pd.MultiIndex.from_arrays([iso["year"].to_numpy(), iso["week"].to_numpy()])
        chosen = pd.DatetimeIndex(pd.Series(eligible, index=keys).groupby(level=[0, 1]).tail(1).to_numpy())
    else:
        keys = pd.MultiIndex.from_arrays([eligible.year, eligible.month])
        chosen = pd.DatetimeIndex(pd.Series(eligible, index=keys).groupby(level=[0, 1]).tail(1).to_numpy())
    return pd.DatetimeIndex(sorted(pd.DatetimeIndex(chosen).unique()))


def run_backtest(
    panel: PricePanel,
    strategy: Strategy,
    config: BacktestConfig | None = None,
    sample: str = "full",
) -> BacktestResult:
    """Run ``strategy`` on ``panel``. Equity is marked at the close."""
    config = config or BacktestConfig()
    index = panel.index
    decisions = decision_dates(index, config.rebalance, strategy.warmup, config.trade_start)
    if len(decisions) == 0:
        raise ValueError(
            f"No rebalance dates for {getattr(strategy, 'name', 'strategy')}. "
            "The sample is shorter than the warmup, or trade_start is too late."
        )
    decision_set = set(pd.Timestamp(d) for d in decisions)
    tickers = panel.tickers
    cash = float(config.initial_capital)
    shares = {ticker: 0.0 for ticker in tickers}
    last_price: dict[str, float] = {}
    missing_streak = {ticker: 0 for ticker in tickers}
    pending: Optional[pd.Series] = None
    equity_values: list[float] = []
    total_costs = 0.0
    turnover = 0.0
    n_rebalances = 0
    decision_log: list[dict] = []
    valuation_events: list[dict] = []

    for i, dt in enumerate(index):
        stamp = pd.Timestamp(dt)
        opens = panel.opens.loc[stamp]
        closes = panel.closes.loc[stamp]
        _remember_prints(opens, last_price)
        if pending is not None:
            equity_before, _ = _mark(cash, shares, opens, last_price)
            cash, shares, traded, cost = _rebalance(
                cash,
                shares,
                pending,
                opens,
                config.commission_bps,
                config.slippage_bps,
                last_price,
            )
            total_costs += cost
            if equity_before > 0:
                turnover += traded / equity_before
            if traded > 1.0:
                n_rebalances += 1
            pending = None

        _remember_prints(closes, last_price)
        for ticker in tickers:
            saw_print = _finite_price(opens, ticker) is not None or _finite_price(closes, ticker) is not None
            missing_streak[ticker] = 0 if saw_print else missing_streak.get(ticker, 0) + 1
        _write_off_delisted(
            shares, last_price, missing_streak, config.delist_after, stamp, valuation_events
        )
        equity_now, stale = _mark(cash, shares, closes, last_price)
        for ticker in stale:
            if shares.get(ticker, 0.0):
                valuation_events.append(
                    {
                        "date": stamp,
                        "ticker": ticker,
                        "action": "stale_mark",
                        "shares": float(shares[ticker]),
                        "price": float(last_price[ticker]),
                    }
                )
        equity_values.append(equity_now)

        if stamp in decision_set and i + 1 < len(index):
            history = panel.through(stamp)
            history_end = pd.Timestamp(history.index.max())
            if history_end > stamp:
                raise RuntimeError(f"History for {stamp.date()} extends to {history_end.date()}")
            weights = strategy.target_weights(stamp, history)
            pending = _sanitize_weights(weights)
            decision_log.append(
                {
                    "decision": stamp,
                    "history_end": history_end,
                    "fill": pd.Timestamp(index[i + 1]),
                    "gross": float(pending.sum()) if len(pending) else 0.0,
                }
            )

    equity = pd.Series(equity_values, index=index, name="equity")
    equity = equity.loc[decisions[0] :]
    metrics = compute_metrics(equity, turnover, total_costs, n_rebalances)
    metrics["strategy"] = strategy.name
    metrics["sample"] = sample
    metrics["n_stale_marks"] = int(sum(1 for event in valuation_events if event["action"] == "stale_mark"))
    metrics["n_delistings"] = int(sum(1 for event in valuation_events if event["action"] == "delisted"))
    return BacktestResult(
        name=strategy.name,
        sample=sample,
        equity=equity,
        returns=equity.pct_change(),
        metrics=metrics,
        total_costs=total_costs,
        turnover=turnover,
        decision_log=decision_log,
        params=dict(getattr(strategy, "params", {}) or {}),
        valuation_events=valuation_events,
    )


def _sanitize_weights(weights: pd.Series | None) -> pd.Series:
    if weights is None or len(weights) == 0:
        return pd.Series(dtype=float)
    cleaned = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cleaned = cleaned.clip(lower=0.0)
    cleaned = cleaned[cleaned > 0]
    if cleaned.empty:
        return pd.Series(dtype=float)
    gross = float(cleaned.sum())
    if gross > 1.0:
        cleaned = cleaned / gross
    return cleaned


def _finite_price(prices: pd.Series, ticker: str) -> Optional[float]:
    px = prices.get(ticker, np.nan) if hasattr(prices, "get") else np.nan
    if px is None or not np.isfinite(px) or float(px) <= 0:
        return None
    return float(px)


def _remember_prints(prices: pd.Series, last_price: dict[str, float]) -> None:
    for ticker in prices.index:
        px = _finite_price(prices, str(ticker))
        if px is not None:
            last_price[str(ticker)] = px


def _mark(
    cash: float,
    shares: dict[str, float],
    prices: pd.Series,
    last_price: dict[str, float],
) -> tuple[float, list[str]]:
    """Mark the book. A missing print uses the last positive price, not zero.

    Names with no price ever observed are omitted and listed as unresolved.
    Held names marked on a stale print are returned so the caller can log them.
    """
    total = float(cash)
    stale: list[str] = []
    for ticker, qty in shares.items():
        if not qty:
            continue
        px = _finite_price(prices, ticker)
        if px is None:
            px = last_price.get(ticker)
            if px is None:
                continue
            stale.append(ticker)
        total += float(qty) * float(px)
    return total, stale


def _write_off_delisted(
    shares: dict[str, float],
    last_price: dict[str, float],
    missing_streak: dict[str, int],
    delist_after: int,
    stamp: pd.Timestamp,
    events: list[dict],
) -> None:
    """Drop a holding after `delist_after` sessions with no print. Proceeds are zero."""
    if delist_after <= 0:
        return
    for ticker, qty in list(shares.items()):
        if qty and missing_streak.get(ticker, 0) >= delist_after:
            px = float(last_price.get(ticker, 0.0))
            events.append(
                {
                    "date": stamp,
                    "ticker": ticker,
                    "action": "delisted",
                    "shares": float(qty),
                    "price": px,
                    "loss": float(qty * px),
                }
            )
            shares[ticker] = 0.0
            missing_streak[ticker] = 0


def _rebalance(
    cash: float,
    shares: dict[str, float],
    target_weights: pd.Series,
    opens: pd.Series,
    commission_bps: float,
    slippage_bps: float,
    last_price: dict[str, float],
) -> tuple[float, dict[str, float], float, float]:
    """Trade toward ``target_weights`` at this open. Returns cash, shares, notional, dollar cost.

    Holdings with no open stay put. Their last-price value is not spent again
    on names that do have an open.
    """
    commission = commission_bps / 10_000.0
    slippage = slippage_bps / 10_000.0
    equity, _stale = _mark(cash, shares, opens, last_price)
    locked = {
        ticker
        for ticker, qty in shares.items()
        if qty and _finite_price(opens, ticker) is None
    }
    locked_value = sum(shares[ticker] * last_price[ticker] for ticker in locked if ticker in last_price)
    deployable = max(equity - locked_value, 0.0)
    target_shares = {ticker: (shares[ticker] if ticker in locked else 0.0) for ticker in shares}

    if deployable > 0 and len(target_weights):
        tradable = []
        for ticker, weight in target_weights.items():
            if ticker not in shares or ticker in locked:
                continue
            px = _finite_price(opens, ticker)
            if px is None or float(weight) <= 0:
                continue
            tradable.append((ticker, float(weight), px))
        weight_sum = sum(weight for _, weight, _ in tradable)
        if weight_sum > 0:
            gross = min(weight_sum, 1.0)
            scale = gross / weight_sum
            for ticker, weight, px in tradable:
                target_shares[ticker] = scale * weight * deployable / px

    traded = 0.0
    cost = 0.0

    for ticker, qty in list(shares.items()):
        delta = target_shares.get(ticker, 0.0) - qty
        if delta >= -1e-12:
            continue
        px = opens.get(ticker, np.nan)
        if px is None or not np.isfinite(px) or float(px) <= 0:
            continue
        fill = float(px) * (1.0 - slippage)
        sell_qty = -delta
        notional = sell_qty * fill
        fee = notional * commission
        cash += notional - fee
        shares[ticker] = qty - sell_qty
        traded += notional
        cost += fee + sell_qty * float(px) * slippage

    buys: list[tuple[str, float, float, float]] = []
    for ticker, target in target_shares.items():
        delta = target - shares.get(ticker, 0.0)
        if delta <= 1e-12:
            continue
        px = opens.get(ticker, np.nan)
        if px is None or not np.isfinite(px) or float(px) <= 0:
            continue
        fill = float(px) * (1.0 + slippage)
        buys.append((ticker, delta, fill, float(px)))

    required = sum(delta * fill * (1.0 + commission) for _, delta, fill, _ in buys)
    buy_scale = 1.0 if required <= cash + 1e-8 or required <= 0 else cash / required
    for ticker, delta, fill, px in buys:
        buy_qty = delta * buy_scale
        notional = buy_qty * fill
        fee = notional * commission
        cash -= notional + fee
        shares[ticker] = shares.get(ticker, 0.0) + buy_qty
        traded += notional
        cost += fee + buy_qty * px * slippage

    if abs(cash) < 1e-6:
        cash = 0.0
    return cash, shares, traded, cost
