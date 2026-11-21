"""Holding-period labels aligned with next-open execution.

A decision at the close of session t is filled at the next open. The label is
the open-to-open return over the following ``horizon`` sessions, not a same-day
close-to-close return. Building the label uses later opens; it must not be
used as a feature or as a training target until that exit open is observable.
"""

from __future__ import annotations

import pandas as pd


def forward_open_return(opens: pd.Series, horizon: int) -> pd.Series:
    """Return from open[t+1] to open[t+1+horizon], indexed at the decision close t."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1 session")
    entry = opens.shift(-1)
    exit_price = opens.shift(-1 - int(horizon))
    return exit_price / entry - 1.0


def label_overlaps_test(decision_position: int, test_start: int, horizon: int) -> bool:
    """True when the open-to-open label for this decision reaches into the test window.

    Entry is the next session (position + 1). Exit is that entry plus ``horizon``
    sessions. The label is contaminated for training if the exit is on or after
    the first test session.
    """
    exit_position = decision_position + 1 + int(horizon)
    return exit_position >= test_start
