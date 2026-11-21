"""Run the top-N rule, momentum, and buy-and-hold benchmarks on one panel."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence

import pandas as pd

from ..analytics import SECTOR_ETF
from .engine import BacktestConfig, BacktestResult, run_backtest
from .panel import PricePanel
from .strategies import (
    MOMENTUM_GRID,
    BuyAndHold,
    MomentumStrategy,
    ParamSchedule,
    TopNThemeStrategy,
)


@dataclass
class WalkForward:
    train_days: int = 504
    test_days: int = 126
    step_days: int = 126

    def __post_init__(self) -> None:
        if self.train_days < 1 or self.test_days < 1:
            raise ValueError("train_days and test_days must be positive")
        if self.step_days != self.test_days:
            raise ValueError(
                "step_days must equal test_days so out-of-sample windows tile the record "
                "without overlap or a gap"
            )


@dataclass
class Comparison:
    results: list[BacktestResult]
    summary: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    tuning: list[dict] = field(default_factory=list)


def iter_windows(
    index: pd.DatetimeIndex,
    walk: WalkForward,
    include_partial: bool = False,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Non-overlapping test windows.

    With ``include_partial``, a final shorter test window is kept when a full
    train is available but the remaining sessions are shorter than ``test_days``.
    That keeps the latest bars in the out-of-sample record.
    """
    windows = []
    start = 0
    while start + walk.train_days + walk.test_days <= len(index):
        train = index[start : start + walk.train_days]
        test = index[start + walk.train_days : start + walk.train_days + walk.test_days]
        windows.append((train, test))
        start += walk.step_days
    if include_partial and start + walk.train_days + 1 < len(index):
        train = index[start : start + walk.train_days]
        test = index[start + walk.train_days :]
        if len(train) == walk.train_days and len(test) >= 2:
            windows.append((train, test))
    return windows


def select_momentum_params(
    panel: PricePanel,
    tickers: Sequence[str],
    grid: Sequence[tuple[int, int]],
    config: BacktestConfig,
    trade_start: pd.Timestamp,
    train_end: pd.Timestamp,
    strategy_kwargs: Optional[dict] = None,
) -> tuple[tuple[int, int], list[dict]]:
    """Pick a momentum pair by in-sample Sharpe. Prices after ``train_end`` are dropped."""
    if not grid:
        raise ValueError("momentum grid is empty")
    kwargs = dict(strategy_kwargs or {})
    kwargs.pop("lookback", None)
    kwargs.pop("skip", None)
    kwargs["sizing"] = "equal"
    sliced = panel.through(pd.Timestamp(train_end))
    best_pair = (int(grid[0][0]), int(grid[0][1]))
    best_score = float("-inf")
    rows = []
    for lookback, skip in grid:
        strategy = MomentumStrategy(tickers, lookback=int(lookback), skip=int(skip), **kwargs)
        train_config = replace(config, trade_start=pd.Timestamp(trade_start))
        try:
            result = run_backtest(sliced, strategy, train_config, sample="in_sample")
            score = result.metrics["sharpe"]
        except ValueError:
            score = float("nan")
        usable = float(score) if score is not None and pd.notna(score) else float("-inf")
        rows.append(
            {
                "lookback": int(lookback),
                "skip": int(skip),
                "train_sharpe": None if usable == float("-inf") else usable,
                "train_end": str(pd.Timestamp(train_end).date()),
                "trade_start": str(pd.Timestamp(trade_start).date()),
            }
        )
        if usable > best_score:
            best_score = usable
            best_pair = (int(lookback), int(skip))
    return best_pair, rows


def compare_strategies(
    panel: PricePanel,
    themes: Mapping[str, Sequence[str]],
    config: Optional[BacktestConfig] = None,
    benchmark: str = "SPY",
    sector_etfs: Optional[Mapping[str, str]] = None,
    oos_start: Optional[str] = None,
    walk_forward: Optional[WalkForward] = None,
    momentum_grid: Sequence[tuple[int, int]] = MOMENTUM_GRID,
    momentum_sizing: str = "both",
    top_n: int = 5,
    earnings: Optional[Mapping[str, pd.Series]] = None,
) -> Comparison:
    """Compare top-N, momentum, SPY, and each theme ETF.

    When ``oos_start`` or ``walk_forward`` is set, the rows labeled
    ``out_of_sample`` are the performance record. Momentum's lookback is chosen
    on the preceding train window only. ``sample=full`` is used only when no
    holdout is requested.
    """
    if momentum_sizing not in {"equal", "vol", "both"}:
        raise ValueError("momentum_sizing must be equal, vol, or both")
    if oos_start and walk_forward:
        raise ValueError("Pass either oos_start or walk_forward, not both")
    config = config or BacktestConfig()
    etfs = {key.lower(): value for key, value in (sector_etfs or SECTOR_ETF).items()}
    themes = {key.lower(): list(values) for key, values in themes.items()}
    constituents = [ticker for names in themes.values() for ticker in names if ticker in panel.closes.columns]
    notes = [
        "Top-N equal-weights the union of each theme's top names.",
        "The earnings-growth filter is off unless a point-in-time earnings series is supplied.",
        "Buy-and-hold benchmarks are traded once, at the next open after the first decision close.",
    ]
    if earnings is None:
        notes.append(
            "No point-in-time earnings were supplied, so last-quarter earnings did not filter the top-N book."
        )

    sizing_modes = ["equal", "vol"] if momentum_sizing == "both" else [momentum_sizing]
    momentum_kwargs = {"top_n": top_n}

    if walk_forward is not None:
        return _run_walk_forward(
            panel, themes, etfs, constituents, benchmark, config, walk_forward,
            momentum_grid, sizing_modes, top_n, earnings, notes, momentum_kwargs,
        )
    if oos_start is not None:
        return _run_single_split(
            panel, themes, etfs, constituents, benchmark, config, pd.Timestamp(oos_start),
            momentum_grid, sizing_modes, top_n, earnings, notes, momentum_kwargs,
        )

    notes.append("No holdout was requested. Metrics are in-sample for the whole downloaded history.")
    strategies = _fixed_strategies(
        themes, etfs, constituents, benchmark, panel, top_n, earnings, sizing_modes,
        lookback=252, skip=21,
    )
    results = [run_backtest(panel, strategy, config, sample="full") for strategy in strategies]
    return Comparison(results=results, summary=_summary(results), notes=notes)


def _run_single_split(
    panel, themes, etfs, constituents, benchmark, config, oos_start, grid, sizing_modes, top_n, earnings, notes, momentum_kwargs,
) -> Comparison:
    index = panel.index
    train_dates = index[index < oos_start]
    test_dates = index[index >= oos_start]
    if len(train_dates) < 2 or len(test_dates) < 2:
        raise ValueError("oos_start leaves an empty train or test sample")
    lookback, skip = 252, 21
    tuning: list[dict] = []
    if constituents and grid:
        pair, rows = select_momentum_params(
            panel, constituents, grid, config,
            trade_start=train_dates[0], train_end=train_dates[-1],
            strategy_kwargs=momentum_kwargs,
        )
        lookback, skip = pair
        tuning.extend(rows)
        notes.append(
            f"Momentum lookback was selected on data through {pd.Timestamp(train_dates[-1]).date()} "
            f"and frozen at {lookback}-{skip} for the out-of-sample run."
        )
    strategies = _fixed_strategies(
        themes, etfs, constituents, benchmark, panel, top_n, earnings, sizing_modes, lookback, skip,
    )
    results = []
    train_panel = panel.through(train_dates[-1])
    for strategy in strategies:
        results.append(run_backtest(train_panel, strategy, replace(config, trade_start=None), sample="in_sample"))
        results.append(
            run_backtest(panel, strategy, replace(config, trade_start=oos_start), sample="out_of_sample")
        )
    notes.append("Read sample=out_of_sample as the performance record. sample=in_sample includes the tuning window.")
    return Comparison(results=results, summary=_summary(results), notes=notes, tuning=tuning)


def _run_walk_forward(
    panel, themes, etfs, constituents, benchmark, config, walk, grid, sizing_modes, top_n, earnings, notes, momentum_kwargs,
) -> Comparison:
    windows = iter_windows(panel.index, walk, include_partial=True)
    if not windows:
        raise ValueError("History is shorter than train_days + test_days")
    tuning: list[dict] = []
    schedules: dict[str, list] = {mode: [] for mode in sizing_modes}
    for train, test in windows:
        lookback, skip = 252, 21
        if constituents and grid:
            pair, rows = select_momentum_params(
                panel, constituents, grid, config,
                trade_start=train[0], train_end=train[-1],
                strategy_kwargs=momentum_kwargs,
            )
            lookback, skip = pair
            for row in rows:
                row["test_start"] = str(pd.Timestamp(test[0]).date())
                row["test_end"] = str(pd.Timestamp(test[-1]).date())
                row["selected"] = row["lookback"] == lookback and row["skip"] == skip
            tuning.extend(rows)
        for mode in sizing_modes:
            strategy = MomentumStrategy(
                constituents, lookback=lookback, skip=skip, sizing=mode, top_n=top_n,
                name=f"momentum_{mode}",
            )
            schedules[mode].append((test[0], test[-1], strategy))
    trade_start = windows[0][1][0]
    trade_end = windows[-1][1][-1]
    oos_panel = panel.through(trade_end)
    strategies = []
    strategies.append(
        TopNThemeStrategy(themes, etfs, benchmark=benchmark, n=top_n, earnings=earnings, name="top_n")
    )
    for mode, segments in schedules.items():
        if segments and constituents:
            strategies.append(ParamSchedule(segments, name=f"momentum_{mode}"))
    strategies.extend(_benchmarks(benchmark, themes, etfs, oos_panel))
    results = [
        run_backtest(oos_panel, strategy, replace(config, trade_start=trade_start), sample="out_of_sample")
        for strategy in strategies
    ]
    notes.append(
        f"Walk-forward used {len(windows)} non-overlapping test windows "
        f"({walk.train_days} in-sample sessions, then {walk.test_days} out-of-sample; "
        "the last window is shorter if that is all that remains). Reported metrics are one continuous "
        "out-of-sample account from the first test window through the last available bar. "
        "The momentum lookback for each window was chosen using prices only through the preceding train end."
    )
    return Comparison(results=results, summary=_summary(results), notes=notes, tuning=tuning)


def _fixed_strategies(themes, etfs, constituents, benchmark, panel, top_n, earnings, sizing_modes, lookback, skip):
    strategies = [
        TopNThemeStrategy(themes, etfs, benchmark=benchmark, n=top_n, earnings=earnings, name="top_n")
    ]
    for mode in sizing_modes:
        if constituents:
            strategies.append(
                MomentumStrategy(
                    constituents, lookback=lookback, skip=skip, sizing=mode, top_n=top_n,
                    name=f"momentum_{mode}",
                )
            )
    strategies.extend(_benchmarks(benchmark, themes, etfs, panel))
    return strategies


def _benchmarks(benchmark, themes, etfs, panel) -> list[BuyAndHold]:
    names = []
    if benchmark in panel.closes.columns:
        names.append(benchmark)
    for theme in themes:
        etf = etfs.get(theme)
        if etf and etf in panel.closes.columns and etf not in names:
            names.append(etf)
    return [BuyAndHold(ticker) for ticker in names]


def _summary(results: Sequence[BacktestResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = dict(result.metrics)
        row["strategy"] = result.name
        row["sample"] = result.sample
        rows.append(row)
    columns = [
        "strategy", "sample", "start", "end", "total_return", "cagr", "ann_vol", "sharpe",
        "sortino", "max_drawdown", "turnover", "ann_turnover", "hit_rate", "total_costs",
        "ending_equity", "n_rebalances",
    ]
    frame = pd.DataFrame(rows)
    ordered = [col for col in columns if col in frame.columns]
    extra = [col for col in frame.columns if col not in ordered]
    return frame[ordered + extra]
