"""Classify a matured prediction from observed returns, not from a story.

A market or sector decline is recorded only when that return is negative.
Overestimated return means the realized net return is below the archived
forecast. These flags can occur together. They are descriptions of the
numbers, not causes.
"""

from __future__ import annotations


def classify_outcome(
    status: str,
    predicted_return: float,
    realized_net: float,
    benchmark_return: float,
    sector_return: float | None = None,
) -> list[str]:
    if status == "pending":
        return ["pending"]
    if status not in {"correct", "incorrect"}:
        return ["unresolved"]
    flags = []
    if realized_net + 1e-12 < predicted_return:
        flags.append("overestimated_return")
    if benchmark_return < 0:
        flags.append("market_decline")
    if sector_return is not None and sector_return < 0:
        flags.append("sector_underperformance")
    if status == "incorrect" and not flags:
        flags.append("stock_specific_or_unclassified")
    if status == "correct" and not flags:
        flags.append("met_or_exceeded_forecast")
    return flags
