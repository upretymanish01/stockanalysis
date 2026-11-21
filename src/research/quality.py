"""Describe holes in a downloaded price panel. This does not fill them."""

from __future__ import annotations

import pandas as pd

from ..backtest.panel import PricePanel


def summarize_prices(panel: PricePanel) -> pd.DataFrame:
    rows = []
    for ticker in panel.tickers:
        close = panel.closes[ticker]
        open_ = panel.opens[ticker]
        valid = close.dropna()
        rows.append(
            {
                "ticker": ticker,
                "sessions": int(len(close)),
                "missing_close": int(close.isna().sum()),
                "missing_open": int(open_.isna().sum()),
                "nonpositive_close": int((valid <= 0).sum()),
                "first_valid": None if valid.empty else str(pd.Timestamp(valid.index[0]).date()),
                "last_valid": None if valid.empty else str(pd.Timestamp(valid.index[-1]).date()),
            }
        )
    return pd.DataFrame(rows)
