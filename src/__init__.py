"""Stock analysis package."""

__version__ = "0.1.0"

from .monte_carlo import (
    geometric_brownian_motion,
    simulate_stock_paths,
    summarize_terminal_prices,
    estimate_params_from_history,
)
from .quality_screen import get_quality_metrics, screen_tickers, score_quality
from .statistical_edge import signal_hit_rate, test_momentum_edge

__all__ = [
    "geometric_brownian_motion",
    "simulate_stock_paths",
    "summarize_terminal_prices",
    "estimate_params_from_history",
    "get_quality_metrics",
    "screen_tickers",
    "score_quality",
    "signal_hit_rate",
    "test_momentum_edge",
]
