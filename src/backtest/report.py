"""Equity curve and summary table for a backtest comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .compare import Comparison

SURVIVORSHIP_CAVEAT = (
    "Survivorship bias: this universe is today's listed tickers from FOCUS_CATEGORIES, "
    "as returned by Yahoo Finance. Companies that were delisted, acquired, or went bankrupt "
    "are absent, and a renamed ticker such as XYZ (formerly SQ) may not carry the old symbol's "
    "history. Backtested results are biased upward relative to a point-in-time membership list. "
    "The live screener's earnings-growth filter is not applied unless a point-in-time earnings "
    "series is supplied, because yfinance .info is the current snapshot, not the figure known on "
    "the decision date."
)

EXECUTION_NOTE = (
    "Signals use only bars available at the close of day t. Orders fill at the next session's "
    "open. Slippage adjusts that fill against the trade, and commission is charged in cash. "
    "Both are basis points of traded notional. Turnover is the sum of traded notional divided "
    "by equity at each fill. Sharpe and Sortino use a zero risk-free rate and 252 trading days. "
    "n_rebalances counts fills whose traded notional exceeds $1, so a buy-and-hold benchmark "
    "is one trade rather than every scheduled check."
)


def write_report(comparison: Comparison, output_dir: str | Path, title: str = "Backtest comparison") -> dict[str, str]:
    """Write summary.csv, summary.md, and equity_curve.png. Returns the paths."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    summary_csv = folder / "summary.csv"
    summary_md = folder / "summary.md"
    chart_path = folder / "equity_curve.png"
    tuning_path = folder / "tuning.csv"

    comparison.summary.to_csv(summary_csv, index=False)
    if comparison.tuning:
        pd.DataFrame(comparison.tuning).to_csv(tuning_path, index=False)
    _write_markdown(comparison, summary_md, title)
    _write_chart(comparison, chart_path, title)
    paths = {"summary_csv": str(summary_csv), "summary_md": str(summary_md), "equity_curve": str(chart_path)}
    if comparison.tuning:
        paths["tuning_csv"] = str(tuning_path)
    return paths


def _write_markdown(comparison: Comparison, path: Path, title: str) -> None:
    frame = comparison.summary
    date_start = frame["start"].min() if "start" in frame.columns and not frame.empty else ""
    date_end = frame["end"].max() if "end" in frame.columns and not frame.empty else ""
    samples = ", ".join(frame["sample"].astype(str).unique()) if "sample" in frame.columns else ""
    lines = [
        f"# {title}",
        "",
        f"**Sessions in the table:** {date_start} to {date_end}",
        f"**Samples:** {samples}",
        "",
        "Rows with `sample=out_of_sample` are the performance record when a holdout or "
        "walk-forward split was used. `sample=in_sample` is the tuning window. "
        "`sample=full` means no holdout was requested.",
        "",
        "## Execution",
        "",
        EXECUTION_NOTE,
        "",
        "## Survivorship and data limits",
        "",
        SURVIVORSHIP_CAVEAT,
        "",
        "## Summary",
        "",
        _markdown_table(frame),
        "",
        "## Notes",
        "",
    ]
    for note in comparison.notes:
        lines.append(f"- {note}")
    if comparison.tuning:
        lines.extend(["", "Parameter choices for each train window are in `tuning.csv`.", ""])
    path.write_text("\n".join(lines) + "\n")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No results._"
    display = frame.copy()
    percent_cols = ["total_return", "cagr", "ann_vol", "max_drawdown", "hit_rate"]
    number_cols = ["sharpe", "sortino", "turnover", "ann_turnover"]
    for col in percent_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda value: _format_percent(value))
    for col in number_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda value: _format_number(value))
    if "total_costs" in display.columns:
        display["total_costs"] = display["total_costs"].map(lambda value: _format_number(value, digits=2))
    if "ending_equity" in display.columns:
        display["ending_equity"] = display["ending_equity"].map(lambda value: _format_number(value, digits=2))
    header = "| " + " | ".join(str(col) for col in display.columns) + " |"
    rule = "| " + " | ".join("---" for _ in display.columns) + " |"
    body = [
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body])


def _write_chart(comparison: Comparison, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for result in comparison.results:
        if result.equity.empty:
            continue
        rebased = result.equity / float(result.equity.iloc[0])
        ax.plot(rebased.index, rebased.to_numpy(), label=f"{result.name} ({result.sample})")
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _format_percent(value) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{100.0 * number:.2f}%"


def _format_number(value, digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _as_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _cell(value) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|")
