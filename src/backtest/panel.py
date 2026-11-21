"""Aligned open/close/volume panels for the backtester."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import pandas as pd

from ..data_fetcher import fetch_price_history

logger = logging.getLogger(__name__)

_OHLCV = ("Open", "High", "Low", "Close", "Volume")


@dataclass
class PricePanel:
    """Columns are tickers. Rows share one trading calendar.

    ``through`` returns a read-only view of sessions up to and including
    ``as_of``. Strategies must not write into that view.
    """

    opens: pd.DataFrame
    closes: pd.DataFrame
    volumes: pd.DataFrame

    def __post_init__(self) -> None:
        if not self.opens.index.equals(self.closes.index) or not self.opens.index.equals(self.volumes.index):
            raise ValueError("Open, close, and volume panels must share an index")
        if list(self.opens.columns) != list(self.closes.columns) or list(self.opens.columns) != list(self.volumes.columns):
            raise ValueError("Open, close, and volume panels must share columns")

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.closes.index)

    @property
    def tickers(self) -> list[str]:
        return [str(c) for c in self.closes.columns]

    def through(self, as_of: pd.Timestamp) -> "PricePanel":
        """Sessions with timestamps less than or equal to ``as_of``."""
        end = int(self.index.searchsorted(pd.Timestamp(as_of), side="right"))
        return PricePanel(
            opens=self.opens.iloc[:end],
            closes=self.closes.iloc[:end],
            volumes=self.volumes.iloc[:end],
        )

    def copy(self) -> "PricePanel":
        return PricePanel(self.opens.copy(), self.closes.copy(), self.volumes.copy())

    def slice(self, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> "PricePanel":
        """Inclusive date clip. ``end`` is the last session to keep."""
        idx = self.index
        if start is not None:
            idx = idx[idx >= pd.Timestamp(start)]
        if end is not None:
            idx = idx[idx <= pd.Timestamp(end)]
        return PricePanel(self.opens.loc[idx], self.closes.loc[idx], self.volumes.loc[idx])


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(_OHLCV))
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    renamed = {col: str(col).strip().title() for col in out.columns}
    out = out.rename(columns=renamed)
    for col in _OHLCV:
        if col not in out.columns:
            out[col] = pd.NA
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    out.index = pd.DatetimeIndex(idx).normalize()
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    return out.loc[:, list(_OHLCV)]


def panel_from_frames(
    frames: Mapping[str, pd.DataFrame],
    calendar: Optional[str] = None,
) -> PricePanel:
    """Outer-align ticker frames. ``calendar`` restricts rows to that ticker's dates."""
    if not frames:
        raise ValueError("No price frames to align")
    normalized = {str(ticker): _normalize_ohlcv(frame) for ticker, frame in frames.items()}
    if calendar is not None and calendar in normalized and not normalized[calendar].empty:
        master = normalized[calendar].index
    else:
        master = pd.DatetimeIndex(sorted(set().union(*[frame.index for frame in normalized.values() if len(frame)])))
    if len(master) == 0:
        raise ValueError("Price frames are empty")
    tickers = list(normalized)
    opens = pd.DataFrame({t: normalized[t]["Open"].reindex(master) for t in tickers}, index=master)
    closes = pd.DataFrame({t: normalized[t]["Close"].reindex(master) for t in tickers}, index=master)
    volumes = pd.DataFrame({t: normalized[t]["Volume"].reindex(master) for t in tickers}, index=master)
    opens.index.name = "date"
    closes.index.name = "date"
    volumes.index.name = "date"
    return PricePanel(opens=opens, closes=closes, volumes=volumes)


def panel_from_csv(path: str) -> PricePanel:
    """Load a long CSV with columns date, ticker, open, close, and optional volume."""
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "open", "close"}
    lowered = {str(col).lower(): col for col in df.columns}
    missing = required.difference(lowered)
    if missing:
        raise ValueError(f"Price CSV is missing columns: {sorted(missing)}")
    df = df.rename(columns={lowered[name]: name for name in lowered})
    if "volume" not in df.columns:
        df["volume"] = pd.NA
    frames = {}
    for ticker, sub in df.groupby("ticker"):
        frame = sub.set_index("date").sort_index()
        frames[str(ticker)] = frame.rename(
            columns={"open": "Open", "close": "Close", "volume": "Volume", "high": "High", "low": "Low"}
        )
    return panel_from_frames(frames)


def load_yahoo_panel(
    tickers: Sequence[str],
    start: str,
    end: str,
    calendar: Optional[str] = "SPY",
) -> tuple[PricePanel, list[str]]:
    """Download daily bars. ``end`` is inclusive for the caller and exclusive for yfinance.

    Returns the panel and the tickers that were skipped after a logged fetch failure.
    """
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frames: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for ticker in list(dict.fromkeys(tickers)):
        try:
            df = fetch_price_history(ticker, start=start, end=end_exclusive)
        except Exception:
            skipped.append(ticker)
            continue
        if df is None or df.empty:
            skipped.append(ticker)
            continue
        frames[ticker] = df
    if skipped:
        logger.warning(
            "Skipped %d ticker(s) due to data fetch failure: %s",
            len(skipped),
            ", ".join(skipped),
        )
    if not frames:
        raise ValueError("No price data was downloaded")
    cal = calendar if calendar in frames else None
    return panel_from_frames(frames, calendar=cal), skipped
