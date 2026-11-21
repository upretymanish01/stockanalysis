"""Cross-sectional rows for one holding period. Labels are separate from features."""

from __future__ import annotations

import pandas as pd

from ..backtest.panel import PricePanel
from .features import price_features
from .labels import forward_open_return


def build_learning_frame(
    panel: PricePanel,
    tickers: list[str],
    horizon: int,
    benchmark: str = "SPY",
    round_trip_cost_bps: float = 12.0,
) -> pd.DataFrame:
    """One row per ticker and decision date.

    ``fwd_gross`` is the next-open holding-period return. ``fwd_net`` subtracts
    a flat round-trip cost (entry and exit, each charged ``round_trip_cost_bps / 2``).
    Rows whose exit open does not yet exist are kept with a missing label so a
    ranker can score the latest close; training must drop them.
    """
    if benchmark not in panel.closes.columns:
        raise ValueError(f"Benchmark {benchmark} is not in the panel")
    cost = float(round_trip_cost_bps) / 10_000.0
    frames = []
    bench = panel.closes[benchmark]
    for ticker in tickers:
        if ticker not in panel.closes.columns or ticker == benchmark:
            continue
        features = price_features(panel.closes[ticker], panel.volumes[ticker], bench)
        features["fwd_gross"] = forward_open_return(panel.opens[ticker], horizon)
        features["fwd_net"] = features["fwd_gross"] - cost
        features["ticker"] = ticker
        features["horizon"] = int(horizon)
        frames.append(features.reset_index().rename(columns={features.index.name or "index": "date"}))
    if not frames:
        raise ValueError("No tickers available to build a learning frame")
    # reset_index names the column "date" when the index name is date; otherwise "index".
    out = pd.concat(frames, ignore_index=True)
    if "date" not in out.columns:
        out = out.rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"])
    return out


def training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows whose holding period has fully elapsed and whose core features exist."""
    required = ["mom_252_21", "dist_sma_200", "fwd_net"]
    return frame.dropna(subset=required).copy()
