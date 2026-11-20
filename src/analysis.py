"""Core analysis functions."""

import pandas as pd
from .utils import add_moving_averages


def basic_technical_summary(df: pd.DataFrame) -> dict:
    """
    Generate a simple technical summary.
    """
    if df.empty:
        return {"error": "No data"}

    latest = df.iloc[-1]
    df_with_ma = add_moving_averages(df)

    return {
        "latest_close": float(latest["Close"]),
        "volume": int(latest["Volume"]),
        "sma_20": float(df_with_ma["SMA_20"].iloc[-1]) if not pd.isna(df_with_ma["SMA_20"].iloc[-1]) else None,
        "sma_50": float(df_with_ma["SMA_50"].iloc[-1]) if not pd.isna(df_with_ma["SMA_50"].iloc[-1]) else None,
    }
