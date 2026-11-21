"""Daily backtests for the stock-selection rules in this repo.

Signals are computed from data available at the close of day t and filled at
the next session's open. See ``SURVIVORSHIP_CAVEAT`` before treating a result
as evidence that a rule makes money: the ticker lists are today's survivors.

Performance that follows parameter tuning is the out-of-sample block. A
full-sample run is labeled ``sample=full`` and is not a holdout.
"""

from .compare import compare_strategies
from .engine import BacktestConfig, BacktestResult, run_backtest
from .metrics import compute_metrics
from .panel import PricePanel, panel_from_csv, panel_from_frames
from .report import SURVIVORSHIP_CAVEAT, write_report

__all__ = [
    "SURVIVORSHIP_CAVEAT",
    "BacktestConfig",
    "BacktestResult",
    "PricePanel",
    "compare_strategies",
    "compute_metrics",
    "panel_from_csv",
    "panel_from_frames",
    "run_backtest",
    "write_report",
]
