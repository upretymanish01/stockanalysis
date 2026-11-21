"""Reproducible baseline and purged model report. No order routing.

Example:
    python -m src.research --themes tech,finance,healthcare --start 2016-01-01 --end 2026-11-21
"""

from __future__ import annotations

import argparse
import json
import math

from .study import load_theme_panel, run_study


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Baseline strategies and purged 21/42/63-day models")
    parser.add_argument("--themes", default="tech,finance,healthcare")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-11-21")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--holdout-days", type=int, default=126)
    args = parser.parse_args(argv)
    themes = [part.strip() for part in args.themes.split(",") if part.strip()]
    panel, mapping, skipped = load_theme_panel(themes, args.start, args.end, args.benchmark)
    report = run_study(panel, mapping, benchmark=args.benchmark, holdout_days=args.holdout_days)
    report["download"] = {
        "source": "yfinance adjusted daily OHLC",
        "start": args.start,
        "end": args.end,
        "themes": themes,
        "benchmark": args.benchmark,
        "skipped_tickers": skipped,
    }
    print(json.dumps(_jsonable(report), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
