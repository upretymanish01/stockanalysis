"""Stock analysis package."""

__version__ = "0.1.0"

from .monte_carlo import (
    geometric_brownian_motion,
    simulate_stock_paths,
    summarize_terminal_prices,
    estimate_params_from_history,
)

__all__ = [
    "geometric_brownian_motion",
    "simulate_stock_paths",
    "summarize_terminal_prices",
    "estimate_params_from_history",
]
