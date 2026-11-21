"""Run the strategy baseline and the purged model comparison on one panel.

The command prints what the code calculated. It does not place orders.
"""

from __future__ import annotations

import json
from typing import Sequence

import pandas as pd

from ..analytics import FOCUS_CATEGORIES, SECTOR_ETF
from ..backtest.compare import WalkForward, compare_strategies
from ..backtest.engine import BacktestConfig
from ..backtest.metrics import compute_metrics
from ..backtest.panel import PricePanel, load_yahoo_panel
from .dataset import build_learning_frame
from .models import evaluate_horizon, mean_daily_feature_ics
from .quality import summarize_prices
from .splits import holdout_start, pre_holdout_train_dates


def run_study(
    panel: PricePanel,
    themes: dict[str, Sequence[str]],
    benchmark: str = "SPY",
    holdout_days: int = 126,
    round_trip_cost_bps: float = 12.0,
) -> dict:
    config = BacktestConfig(rebalance="monthly", commission_bps=1.0, slippage_bps=5.0)
    comparison = compare_strategies(
        panel,
        themes,
        config=config,
        benchmark=benchmark,
        sector_etfs=SECTOR_ETF,
        walk_forward=WalkForward(train_days=504, test_days=126, step_days=126),
    )
    cutoff = holdout_start(panel.index, holdout_days)
    final_window = []
    for result in comparison.results:
        sliced = result.equity.loc[result.equity.index >= cutoff]
        if len(sliced) < 3:
            continue
        metrics = compute_metrics(sliced, turnover=float("nan"), total_costs=float("nan"), n_rebalances=0)
        metrics["strategy"] = result.name
        metrics["sample"] = "final_126_sessions_of_walk_forward"
        final_window.append(metrics)
    constituents = [
        ticker
        for names in themes.values()
        for ticker in names
        if ticker in panel.closes.columns and ticker != benchmark
    ]
    horizons = []
    for horizon in (21, 42, 63):
        frame = build_learning_frame(
            panel,
            constituents,
            horizon,
            benchmark=benchmark,
            round_trip_cost_bps=round_trip_cost_bps,
        )
        report = evaluate_horizon(
            frame,
            panel.index,
            horizon,
            holdout_days=holdout_days,
            round_trip_cost_bps=round_trip_cost_bps,
        )
        develop_dates = pre_holdout_train_dates(panel.index, horizon, holdout_days)
        holdout_dates = panel.index[panel.index >= cutoff]
        horizons.append(
            {
                "horizon": report.horizon,
                "selected_model": report.selected_model,
                "validation": report.validation,
                "holdout": report.holdout,
                "feature_ic_development": mean_daily_feature_ics(frame, develop_dates),
                "feature_ic_holdout": mean_daily_feature_ics(frame, holdout_dates),
                "notes": report.notes,
            }
        )
    return {
        "price_quality": summarize_prices(panel).to_dict(orient="records"),
        "baseline": comparison.summary.to_dict(orient="records"),
        "baseline_notes": comparison.notes,
        "final_window": final_window,
        "horizons": horizons,
        "holdout_start": str(pd.Timestamp(cutoff).date()),
    }


def load_theme_panel(themes: Sequence[str], start: str, end: str, benchmark: str = "SPY"):
    mapping = {}
    tickers = [benchmark]
    for theme in themes:
        key = theme.lower()
        if key not in FOCUS_CATEGORIES:
            raise ValueError(f"Unknown theme {theme}")
        mapping[key] = list(FOCUS_CATEGORIES[key])
        tickers.extend(FOCUS_CATEGORIES[key])
        etf = SECTOR_ETF.get(key)
        if etf:
            tickers.append(etf)
    panel, skipped = load_yahoo_panel(tickers, start=start, end=end, calendar=benchmark)
    return panel, mapping, skipped
