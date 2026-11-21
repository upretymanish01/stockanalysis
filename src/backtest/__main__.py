"""Command line entry point: python -m src.backtest --help"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from ..analytics import FOCUS_CATEGORIES, SECTOR_ETF
from .compare import WalkForward, compare_strategies
from .engine import BacktestConfig
from .panel import load_yahoo_panel, panel_from_csv
from .report import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest the repo's top-5-per-theme rule against 12-1 momentum "
            "and buy-and-hold benchmarks. Signals use the close; fills use the next open."
        )
    )
    parser.add_argument(
        "--themes",
        default="tech,finance,healthcare",
        help="Comma-separated FOCUS_CATEGORIES keys. Default: tech,finance,healthcare",
    )
    parser.add_argument("--benchmark", default="SPY", help="Buy-and-hold market benchmark. Default: SPY")
    parser.add_argument("--start", default="2018-01-01", help="First session to download or keep (inclusive)")
    parser.add_argument("--end", default=None, help="Last session to download or keep (inclusive)")
    parser.add_argument(
        "--prices",
        default=None,
        help="Optional long CSV (date,ticker,open,close,volume) instead of Yahoo Finance",
    )
    parser.add_argument("--rebalance", choices=["daily", "weekly", "monthly"], default="monthly")
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--momentum-sizing",
        choices=["equal", "vol", "both"],
        default="both",
        help="Equal weight, inverse-vol with a 15%% vol cap, or both",
    )
    parser.add_argument(
        "--oos-start",
        default=None,
        help="Holdout date. Momentum's lookback is chosen only on earlier data.",
    )
    parser.add_argument("--walk-forward", action="store_true", help="Tile non-overlapping out-of-sample windows")
    parser.add_argument("--train-days", type=int, default=504)
    parser.add_argument("--test-days", type=int, default=126)
    parser.add_argument("--output", default="output/backtest")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    themes = _themes_from_arg(args.themes)
    config = BacktestConfig(
        initial_capital=args.capital,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        rebalance=args.rebalance,
    )
    panel = _load_panel(args, themes)
    if args.start or args.end:
        panel = panel.slice(args.start, args.end)
    walk = None
    if args.walk_forward:
        walk = WalkForward(train_days=args.train_days, test_days=args.test_days, step_days=args.test_days)
    comparison = compare_strategies(
        panel,
        themes,
        config=config,
        benchmark=args.benchmark,
        sector_etfs=SECTOR_ETF,
        oos_start=args.oos_start,
        walk_forward=walk,
        momentum_sizing=args.momentum_sizing,
        top_n=args.top_n,
    )
    paths = write_report(comparison, args.output, title="Stock-selection backtest")
    print(comparison.summary.to_string(index=False))
    print()
    for note in comparison.notes:
        print(f"- {note}")
    print()
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 0


def _themes_from_arg(text: str) -> dict[str, list[str]]:
    themes = {}
    for raw in text.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        if key not in FOCUS_CATEGORIES:
            known = ", ".join(sorted(FOCUS_CATEGORIES))
            raise SystemExit(f"Unknown theme '{key}'. Known themes: {known}")
        themes[key] = list(FOCUS_CATEGORIES[key])
    if not themes:
        raise SystemExit("Pass at least one theme")
    return themes


def _load_panel(args, themes: dict[str, list[str]]):
    if args.prices:
        panel = panel_from_csv(args.prices)
        missing = [ticker for names in themes.values() for ticker in names if ticker not in panel.closes.columns]
        if args.benchmark not in panel.closes.columns:
            missing.append(args.benchmark)
        if missing:
            logging.getLogger(__name__).warning(
                "CSV is missing %d ticker(s): %s", len(missing), ", ".join(missing)
            )
        return panel
    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = [args.benchmark]
    for theme, names in themes.items():
        tickers.extend(names)
        etf = SECTOR_ETF.get(theme)
        if etf:
            tickers.append(etf)
    panel, _skipped = load_yahoo_panel(tickers, start=args.start, end=end, calendar=args.benchmark)
    return panel


if __name__ == "__main__":
    sys.exit(main())
