"""Purged walk-forward splits and a final holdout.

Overlapping open-to-open labels leak if a training row's exit falls inside the
test window. Those rows are purged. The holdout is removed before any fold is
built, so it cannot influence model choice or the probability threshold.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from .labels import label_overlaps_test


def holdout_start(dates: pd.DatetimeIndex, holdout_days: int) -> pd.Timestamp:
    if holdout_days < 1:
        raise ValueError("holdout_days must be positive")
    if holdout_days >= len(dates):
        raise ValueError("holdout is longer than the sample")
    return pd.Timestamp(dates[-int(holdout_days)])


def purged_walk_forward(
    dates: pd.DatetimeIndex,
    horizon: int,
    test_days: int = 126,
    holdout_days: int = 126,
    min_train_days: int = 252,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield ``(train_dates, test_dates)`` on the pre-holdout sample.

    Train dates are session timestamps whose labels do not reach the test
    window. Test dates lie entirely before the holdout.
    """
    if test_days < 1 or min_train_days < 1:
        raise ValueError("test_days and min_train_days must be positive")
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    cutoff = holdout_start(dates, holdout_days)
    usable = dates[dates < cutoff]
    start = min_train_days
    while start + test_days <= len(usable):
        test = usable[start : start + test_days]
        test_pos = int(dates.get_loc(test[0]))
        train = [
            stamp
            for stamp in usable[:start]
            if not label_overlaps_test(int(dates.get_loc(stamp)), test_pos, horizon)
        ]
        if train:
            yield pd.DatetimeIndex(train), pd.DatetimeIndex(test)
        start += test_days


def pre_holdout_train_dates(
    dates: pd.DatetimeIndex,
    horizon: int,
    holdout_days: int,
) -> pd.DatetimeIndex:
    """Dates whose labels are fully known before the holdout starts."""
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    cutoff = holdout_start(dates, holdout_days)
    test_pos = int(dates.get_loc(cutoff))
    kept = [
        stamp
        for stamp in dates
        if stamp < cutoff and not label_overlaps_test(int(dates.get_loc(stamp)), test_pos, horizon)
    ]
    return pd.DatetimeIndex(kept)
