"""Analytics module: retrieve data and produce summary analytics + visualizations.

Combines price history, basic stats, quality metrics, and simple plots.
Designed so results can be easily saved into analysis-results/ dated folders.

Includes multi-period comparison so you can capture data once and compare
2/3/4/5 days, 1 week, 1 month, 3 months, 6 months, and years.

Every analysis run saves both Markdown (human) and CSV (viewer / reuse).
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from .data_fetcher import fetch_price_history
from .quality_screen import get_quality_metrics


def get_price_analytics(ticker: str, period: str = "1y") -> Dict:
    """
    Fetch price history and compute key analytics.

    Returns dictionary with summary stats and the full DataFrame.
    """
    df = fetch_price_history(ticker, period=period)
    if df.empty:
        return {"ticker": ticker, "error": "No data"}

    close = df["Close"]
    returns = close.pct_change().dropna()

    analytics = {
        "ticker": ticker,
        "period": period,
        "start_date": str(df.index[0].date()),
        "end_date": str(df.index[-1].date()),
        "latest_close": float(close.iloc[-1]),
        "total_return": float((close.iloc[-1] / close.iloc[0]) - 1),
        "annualized_return": float(returns.mean() * 252),
        "annualized_volatility": float(returns.std() * (252 ** 0.5)),
        "max_drawdown": float((close / close.cummax() - 1).min()),
        "sharpe_approx": float((returns.mean() * 252) / (returns.std() * (252 ** 0.5))) if returns.std() > 0 else None,
        "avg_volume": float(df["Volume"].mean()),
        "data": df,
    }
    return analytics


def multi_period_returns(
    ticker: str,
    lookbacks: Optional[List[int]] = None,
    base_period: str = "2y",
) -> pd.DataFrame:
    """
    Capture price data once, then compute total returns over multiple lookback windows.

    Default lookbacks (trading days):
    - 2, 3, 4, 5 days
    - 5 (~1 week)
    - 21 (~1 month)
    - 63 (~3 months)
    - 126 (~6 months)
    - 252 (~1 year)
    - 504 (~2 years)

    Returns a DataFrame with period label, days, start price, end price, total return.
    """
    if lookbacks is None:
        lookbacks = [2, 3, 4, 5, 5, 21, 63, 126, 252, 504]

    # Label mapping for readability
    label_map = {
        2: "2d", 3: "3d", 4: "4d", 5: "5d / 1w",
        21: "1m", 63: "3m", 126: "6m",
        252: "1y", 504: "2y",
    }

    df = fetch_price_history(ticker, period=base_period)
    if df.empty or len(df) < max(lookbacks):
        return pd.DataFrame([{"ticker": ticker, "error": "Insufficient data"}])

    close = df["Close"]
    latest = close.iloc[-1]
    rows = []

    for days in lookbacks:
        if days >= len(close):
            continue
        start_price = close.iloc[-days - 1] if days < len(close) else close.iloc[0]
        # More precise: price from N trading days ago
        start_idx = max(0, len(close) - days - 1)
        start_price = close.iloc[start_idx]
        total_ret = (latest / start_price) - 1

        rows.append({
            "ticker": ticker,
            "period_label": label_map.get(days, f"{days}d"),
            "lookback_days": days,
            "start_price": float(start_price),
            "end_price": float(latest),
            "total_return": float(total_ret),
            "total_return_pct": round(total_ret * 100, 2),
        })

    return pd.DataFrame(rows)


def multi_period_universe(
    tickers: List[str],
    lookbacks: Optional[List[int]] = None,
    base_period: str = "2y",
) -> pd.DataFrame:
    """
    Run multi-period returns for multiple tickers.
    Easy to compare side-by-side.
    """
    frames = []
    for t in tickers:
        frames.append(multi_period_returns(t, lookbacks, base_period))
    return pd.concat(frames, ignore_index=True)


def plot_multi_period_returns(
    df: pd.DataFrame,
    title: str = "Multi-Period Returns Comparison",
    save_path: Optional[str] = None,
):
    """
    Bar chart of total returns across periods.
    Works for single ticker or multiple (grouped).
    """
    if df.empty or "error" in df.columns:
        print("No valid data to plot")
        return

    pivot = df.pivot(index="period_label", columns="ticker", values="total_return_pct")
    # Reorder periods if possible
    preferred_order = ["2d", "3d", "4d", "5d / 1w", "1m", "3m", "6m", "1y", "2y"]
    existing = [p for p in preferred_order if p in pivot.index]
    if existing:
        pivot = pivot.reindex(existing)

    ax = pivot.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_title(title)
    ax.set_ylabel("Total Return (%)")
    ax.set_xlabel("Period")
    ax.legend(title="Ticker")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_price_and_returns(analytics: Dict, save_path: Optional[str] = None):
    """Plot price history and cumulative returns."""
    if "error" in analytics:
        print(analytics["error"])
        return

    df = analytics["data"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(df.index, df["Close"], label="Close")
    axes[0].set_title(f"{analytics['ticker']} Price — {analytics['period']}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    cum_ret = (1 + df["Close"].pct_change()).cumprod()
    axes[1].plot(df.index, cum_ret, label="Cumulative Return", color="green")
    axes[1].set_title("Cumulative Return")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def full_ticker_analytics(ticker: str, period: str = "1y") -> Dict:
    """
    Combined price analytics + quality metrics for one ticker.
    """
    price = get_price_analytics(ticker, period)
    quality = get_quality_metrics(ticker)

    summary = {k: v for k, v in price.items() if k != "data"}
    summary["quality"] = quality
    summary["price_df"] = price.get("data")
    return summary


def analytics_for_universe(
    tickers: List[str],
    period: str = "1y",
) -> pd.DataFrame:
    """
    Run full analytics on a list of tickers and return a summary DataFrame.
    """
    rows = []
    for t in tickers:
        try:
            a = full_ticker_analytics(t, period)
            if "error" in a:
                rows.append({"ticker": t, "error": a["error"]})
                continue
            row = {
                "ticker": t,
                "latest_close": a.get("latest_close"),
                "total_return": a.get("total_return"),
                "ann_return": a.get("annualized_return"),
                "ann_vol": a.get("annualized_volatility"),
                "max_dd": a.get("max_drawdown"),
                "sharpe_approx": a.get("sharpe_approx"),
                "roe": a["quality"].get("roe"),
                "trailing_pe": a["quality"].get("trailing_pe"),
                "peg_ratio": a["quality"].get("peg_ratio"),
                "profit_margin": a["quality"].get("profit_margin"),
            }
            rows.append(row)
        except Exception as e:
            rows.append({"ticker": t, "error": str(e)})

    return pd.DataFrame(rows)


def save_analytics_result(
    summary_df: pd.DataFrame,
    title: str,
    model: str = "multi_period_analytics",
    stocks: Optional[List[str]] = None,
    date_str: Optional[str] = None,
    notes: str = "",
) -> Dict[str, str]:
    """
    Save both Markdown summary and CSV data into a dated folder under analysis-results/.

    Every run produces:
    - YYYY-MM-DD/title.md   (human-readable + metadata)
    - YYYY-MM-DD/title.csv  (machine-readable for analysis viewer)

    Returns dict with paths: {"markdown": ..., "csv": ...}
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    folder = Path("analysis-results") / date_str
    folder.mkdir(parents=True, exist_ok=True)

    safe_title = title.lower().replace(" ", "-").replace("/", "-")
    md_path = folder / f"{safe_title}.md"
    csv_path = folder / f"{safe_title}.csv"

    # Determine stocks if not provided
    if stocks is None and "ticker" in summary_df.columns:
        stocks = summary_df["ticker"].unique().tolist()
    stocks_str = ", ".join(stocks) if stocks else "N/A"

    # Save CSV first (clean data for viewer)
    summary_df.to_csv(csv_path, index=False)

    # Markdown with metadata
    content = f"""# Analysis Result — {date_str}

**Title:** {title}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
**Model / Method:** {model}  
**Stocks:** {stocks_str}  
**CSV data:** `{csv_path.name}`  

## Summary Table

{summary_df.to_markdown(index=False) if hasattr(summary_df, 'to_markdown') else summary_df.to_string()}

## Notes
{notes}

## Files
- Markdown: `{md_path.name}`
- CSV: `{csv_path.name}` (use this for the analysis viewer)
"""
    md_path.write_text(content)

    print(f"Saved Markdown: {md_path}")
    print(f"Saved CSV:      {csv_path}")

    return {"markdown": str(md_path), "csv": str(csv_path)}
