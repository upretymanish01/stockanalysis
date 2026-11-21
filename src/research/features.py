"""Price and volume features known at the close.

Every column is a backward-looking transform. Nothing here reads a future bar.
Fundamental ratios are intentionally absent: Yahoo's current snapshot is not
the statement that had been published on the decision date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def price_features(
    close: pd.Series,
    volume: pd.Series,
    benchmark_close: pd.Series,
) -> pd.DataFrame:
    """Feature row for each session in ``close``.

    Momentum windows follow the evidence that intermediate-horizon past returns
    forecast the next several months (Jegadeesh and Titman 1993; Novy-Marx 2012
    on skipping the most recent month). Trend and volatility features describe
    the same history. Benchmark columns are the market state on that close.
    """
    bench = benchmark_close.reindex(close.index)
    returns = close.pct_change()
    bench_returns = bench.pct_change()
    frame = pd.DataFrame(index=close.index)
    frame["mom_21"] = close / close.shift(21) - 1.0
    frame["mom_63"] = close / close.shift(63) - 1.0
    frame["mom_126"] = close / close.shift(126) - 1.0
    frame["mom_252_21"] = close.shift(21) / close.shift(252) - 1.0
    frame["dist_sma_50"] = close / close.rolling(50).mean() - 1.0
    frame["dist_sma_200"] = close / close.rolling(200).mean() - 1.0
    frame["vol_21"] = returns.rolling(21).std()
    frame["vol_63"] = returns.rolling(63).std()
    frame["dd_63"] = close / close.rolling(63).max() - 1.0
    avg_volume = volume.rolling(21).mean()
    frame["volume_ratio_21"] = volume / avg_volume
    frame["rs_63"] = frame["mom_63"] - (bench / bench.shift(63) - 1.0)
    frame["bench_mom_63"] = bench / bench.shift(63) - 1.0
    frame["bench_vol_21"] = bench_returns.rolling(21).std()
    frame["bench_dd_63"] = bench / bench.rolling(63).max() - 1.0
    return frame.replace([np.inf, -np.inf], np.nan)


FEATURE_COLUMNS = [
    "mom_21",
    "mom_63",
    "mom_126",
    "mom_252_21",
    "dist_sma_50",
    "dist_sma_200",
    "vol_21",
    "vol_63",
    "dd_63",
    "volume_ratio_21",
    "rs_63",
    "bench_mom_63",
    "bench_vol_21",
    "bench_dd_63",
]
