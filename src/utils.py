"""Utility functions for stock analysis."""

import pandas as pd


def calculate_returns(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Calculate simple returns."""
    return df[column].pct_change()


def add_moving_averages(
    df: pd.DataFrame,
    windows: list[int] = [20, 50, 200],
    column: str = "Close"
) -> pd.DataFrame:
    """Add simple moving averages."""
    result = df.copy()
    for w in windows:
        result[f"SMA_{w}"] = result[column].rolling(window=w).mean()
    return result
