"""Selection rules compared by the backtester.

Top-N uses the same filters and composite score as ``top_5_by_timeframe``
(40% risk-adjusted return, 20% pattern, 20% relative strength vs SPY, 20%
relative strength vs the theme ETF). Scores are rebuilt from a trailing window
ending at the decision close. The live earnings filter is applied only when a
point-in-time series is passed in; Yahoo's current ``.info`` snapshot is not
used, because that would look ahead.

Momentum is 12-1 month by default: the total return from ``lookback`` sessions
ago to ``skip`` sessions ago, kept only when that return is positive and the
close is above its trend average.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..analytics import _add_composite_score, _noise_score, _pattern_score
from .panel import PricePanel

# Default grid for walk-forward selection: 12-1, 9-1, and 6-1 momentum.
MOMENTUM_GRID = ((252, 21), (189, 21), (126, 21))


def select_top_n(
    df: pd.DataFrame,
    n: int = 5,
    timeframe: str = "1m",
    rank_by: str = "composite_score",
    max_noise: float = 2.0,
    min_pattern: int = 2,
    min_volume: float = 500_000,
    require_positive_earnings: bool = False,
) -> pd.DataFrame:
    """Same filters and ranking as ``top_5_by_timeframe``, with a configurable N.

    ``require_positive_earnings`` defaults to False. The live screener defaults
    to True, but only a point-in-time earnings column is safe inside a backtest.
    """
    if df.empty or "timeframe" not in df.columns:
        return df
    subset = df[df["timeframe"] == timeframe].copy()
    if subset.empty:
        return subset
    if "noise_score" in subset.columns:
        subset = subset[subset["noise_score"].isna() | (subset["noise_score"] <= max_noise)]
    if "pattern_score" in subset.columns:
        subset = subset[subset["pattern_score"].fillna(0) >= min_pattern]
    if "avg_volume" in subset.columns:
        subset = subset[subset["avg_volume"].fillna(0) >= min_volume]
    if require_positive_earnings and "last_quarter_earnings" in subset.columns:
        subset = subset[subset["last_quarter_earnings"].isna() | (subset["last_quarter_earnings"] > 0)]
    subset = _add_composite_score(subset)
    if rank_by not in subset.columns:
        rank_by = "composite_score"
    subset = subset.sort_values(rank_by, ascending=False, na_position="last")
    return subset.head(n).reset_index(drop=True)


class BuyAndHold:
    """Stay fully invested in one ticker after the first eligible close."""

    def __init__(self, ticker: str, name: Optional[str] = None):
        self.ticker = ticker
        self.name = name or f"buy_hold_{ticker}"
        self.warmup = 0
        self.params = {"ticker": ticker}

    def target_weights(self, as_of: pd.Timestamp, history: PricePanel) -> pd.Series:
        if self.ticker not in history.closes.columns or history.closes.empty:
            return pd.Series(dtype=float)
        px = history.closes[self.ticker].iloc[-1]
        if not np.isfinite(px):
            return pd.Series(dtype=float)
        return pd.Series({self.ticker: 1.0})


class Cash:
    name = "cash"
    warmup = 0
    params: dict = {}

    def target_weights(self, as_of: pd.Timestamp, history: PricePanel) -> pd.Series:
        return pd.Series(dtype=float)


class MomentumStrategy:
    """Rank a universe by skipped momentum and keep names in an uptrend."""

    def __init__(
        self,
        tickers: Sequence[str],
        lookback: int = 252,
        skip: int = 21,
        top_n: int = 5,
        sizing: str = "equal",
        trend_window: int = 200,
        vol_lookback: int = 63,
        target_vol: Optional[float] = 0.15,
        name: Optional[str] = None,
    ):
        if lookback <= skip:
            raise ValueError("lookback must be longer than skip")
        if sizing not in {"equal", "vol"}:
            raise ValueError("sizing must be 'equal' or 'vol'")
        if top_n < 1:
            raise ValueError("top_n must be positive")
        self.tickers = list(dict.fromkeys(tickers))
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.top_n = int(top_n)
        self.sizing = sizing
        self.trend_window = int(trend_window)
        self.vol_lookback = int(vol_lookback)
        self.target_vol = target_vol
        self.warmup = max(self.lookback, self.trend_window)
        self.name = name or f"momentum_{lookback}_{skip}_{sizing}"
        self.params = {
            "lookback": self.lookback,
            "skip": self.skip,
            "top_n": self.top_n,
            "sizing": self.sizing,
            "trend_window": self.trend_window,
            "target_vol": self.target_vol,
        }

    def target_weights(self, as_of: pd.Timestamp, history: PricePanel) -> pd.Series:
        available = [ticker for ticker in self.tickers if ticker in history.closes.columns]
        if not available or len(history.closes) <= self.lookback:
            return pd.Series(dtype=float)
        close = history.closes[available]
        momentum = close.iloc[-1 - self.skip] / close.iloc[-1 - self.lookback] - 1.0
        trend = close.rolling(self.trend_window).mean().iloc[-1]
        latest = close.iloc[-1]
        eligible = momentum[(latest > trend) & (momentum > 0)].dropna().sort_values(ascending=False)
        picked = eligible.head(self.top_n)
        if picked.empty:
            return pd.Series(dtype=float)
        if self.sizing == "equal":
            return pd.Series(1.0 / len(picked), index=picked.index)
        return self._vol_weights(close[picked.index])

    def _vol_weights(self, close: pd.DataFrame) -> pd.Series:
        returns = close.pct_change().iloc[-self.vol_lookback :]
        vol = returns.std(ddof=1).replace(0, np.nan)
        inverse = (1.0 / vol).dropna()
        if inverse.empty:
            return pd.Series(1.0 / close.shape[1], index=close.columns)
        weights = inverse / inverse.sum()
        if self.target_vol is None:
            return weights
        aligned = returns[weights.index].dropna(how="any")
        if len(aligned) < 5:
            return weights
        cov = aligned.cov()
        aligned_weights = weights.loc[cov.index].to_numpy(dtype=float)
        port_var = float(aligned_weights @ cov.to_numpy() @ aligned_weights)
        if port_var <= 0:
            return weights
        port_vol = float(np.sqrt(port_var) * np.sqrt(252))
        if port_vol > self.target_vol:
            weights = weights * (self.target_vol / port_vol)
        return weights


class TopNThemeStrategy:
    """Top N names per theme, equal-weighted across the union.

    Relative strength uses a trailing window (default 252 sessions) ending at
    the decision close. The live screener instead uses whatever span was
    downloaded in one shot; both are backward-looking, and the fixed window
    keeps every rebalance on the same horizon.
    """

    def __init__(
        self,
        themes: Mapping[str, Sequence[str]],
        sector_etfs: Mapping[str, str],
        benchmark: str = "SPY",
        n: int = 5,
        lookback_days: int = 21,
        rs_window: int = 252,
        timeframe: str = "1m",
        max_noise: float = 2.0,
        min_pattern: int = 2,
        min_volume: float = 500_000,
        earnings: Optional[Mapping[str, pd.Series]] = None,
        name: str = "top_n",
    ):
        if n < 1:
            raise ValueError("n must be positive")
        self.themes = {key.lower(): list(tickers) for key, tickers in themes.items()}
        self.sector_etfs = {key.lower(): etf for key, etf in sector_etfs.items()}
        self.benchmark = benchmark
        self.n = int(n)
        self.lookback_days = int(lookback_days)
        self.rs_window = int(rs_window)
        self.timeframe = timeframe
        self.max_noise = max_noise
        self.min_pattern = min_pattern
        self.min_volume = min_volume
        self.earnings = earnings
        self.warmup = max(200, self.rs_window, self.lookback_days + 1)
        self.name = name
        self.params = {
            "n": self.n,
            "lookback_days": self.lookback_days,
            "rs_window": self.rs_window,
            "timeframe": self.timeframe,
            "earnings_filter": earnings is not None,
        }

    def target_weights(self, as_of: pd.Timestamp, history: PricePanel) -> pd.Series:
        selected: list[str] = []
        for theme, tickers in self.themes.items():
            rows = []
            sector = self.sector_etfs.get(theme, self.benchmark)
            for ticker in tickers:
                if ticker not in history.closes.columns:
                    continue
                row = self._score_row(ticker, history, sector, as_of)
                if row is not None:
                    rows.append(row)
            if not rows:
                continue
            frame = pd.DataFrame(rows)
            picked = select_top_n(
                frame,
                n=self.n,
                timeframe=self.timeframe,
                max_noise=self.max_noise,
                min_pattern=self.min_pattern,
                min_volume=self.min_volume,
                require_positive_earnings=self.earnings is not None,
            )
            selected.extend(str(ticker) for ticker in picked.get("ticker", []))
        unique = list(dict.fromkeys(selected))
        if not unique:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(unique), index=unique)

    def _score_row(
        self,
        ticker: str,
        history: PricePanel,
        sector: str,
        as_of: pd.Timestamp,
    ) -> Optional[dict]:
        close = history.closes[ticker].dropna()
        if len(close) <= self.lookback_days:
            return None
        start = close.iloc[-1 - self.lookback_days]
        latest = close.iloc[-1]
        if not np.isfinite(start) or start <= 0 or not np.isfinite(latest):
            return None
        total_ret = float(latest / start - 1.0)
        returns = close.pct_change().dropna()
        window = returns.tail(self.lookback_days) if self.lookback_days < len(returns) else returns
        vol = float(window.std()) if len(window) else float("nan")
        risk_adj = float(total_ret / vol) if vol and vol > 0 else float("nan")
        volume = history.volumes[ticker].reindex(close.index).tail(self.rs_window)
        avg_volume = float(volume.mean()) if volume.notna().any() else float("nan")
        row = {
            "ticker": ticker,
            "timeframe": self.timeframe,
            "risk_adjusted_score": risk_adj,
            "noise_score": _noise_score(close),
            "pattern_score": _pattern_score(close),
            "relative_strength_vs_spy": _relative_return(close, _series(history, self.benchmark), self.rs_window),
            "relative_strength_vs_sector": _relative_return(close, _series(history, sector), self.rs_window),
            "avg_volume": avg_volume,
        }
        if self.earnings is not None:
            row["last_quarter_earnings"] = _earnings_as_of(self.earnings.get(ticker), as_of)
        return row


class ParamSchedule:
    """Apply a strategy that was chosen before the interval that contains ``as_of``."""

    def __init__(self, segments: Sequence[tuple[pd.Timestamp, pd.Timestamp, object]], name: str):
        self.segments = [
            (pd.Timestamp(start), pd.Timestamp(end), strategy) for start, end, strategy in segments
        ]
        self.name = name
        self.warmup = max((int(getattr(strategy, "warmup", 0)) for _, _, strategy in self.segments), default=0)
        self.params = {"segments": len(self.segments)}

    def target_weights(self, as_of: pd.Timestamp, history: PricePanel) -> pd.Series:
        stamp = pd.Timestamp(as_of)
        for start, end, strategy in self.segments:
            if start <= stamp <= end:
                return strategy.target_weights(stamp, history)
        return pd.Series(dtype=float)


def _series(history: PricePanel, ticker: str) -> Optional[pd.Series]:
    if ticker not in history.closes.columns:
        return None
    return history.closes[ticker]


def _relative_return(stock: pd.Series, benchmark: Optional[pd.Series], window: int) -> float:
    if benchmark is None:
        return float("nan")
    aligned = pd.concat([stock.rename("stock"), benchmark.rename("bench")], axis=1).dropna()
    if len(aligned) <= window:
        return float("nan")
    stock_ret = float(aligned["stock"].iloc[-1] / aligned["stock"].iloc[-1 - window] - 1.0)
    bench_ret = float(aligned["bench"].iloc[-1] / aligned["bench"].iloc[-1 - window] - 1.0)
    return stock_ret - bench_ret


def _earnings_as_of(series: Optional[pd.Series], as_of: pd.Timestamp) -> float:
    if series is None or series.empty:
        return float("nan")
    known = series.loc[:pd.Timestamp(as_of)].dropna()
    if known.empty:
        return float("nan")
    return float(known.iloc[-1])
