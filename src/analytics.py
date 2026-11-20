"""Analytics module: retrieve data and produce summary analytics + visualizations.

Combines price history, basic stats, quality metrics, and simple plots.
Designed so results can be easily saved into analysis-results/ dated folders.
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

    # Remove the large DataFrame from the returned summary if desired
    summary = {k: v for k, v in price.items() if k != "data"}
    summary["quality"] = quality
    summary["price_df"] = price.get("data")  # keep available if needed
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
    date_str: Optional[str] = None,
    notes: str = "",
) -> str:
    """
    Save a Markdown analysis result into a dated folder under analysis-results/.

    Returns the path of the created file.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    folder = Path("analysis-results") / date_str
    folder.mkdir(parents=True, exist_ok=True)

    safe_title = title.lower().replace(" ", "-").replace("/", "-")
    filepath = folder / f"{safe_title}.md"

    content = f"""# Analysis Result — {date_str}

**Title:** {title}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  

## Summary Table

{summary_df.to_markdown(index=False) if hasattr(summary_df, 'to_markdown') else summary_df.to_string()}

## Notes
{notes}

## Raw DataFrame (CSV-style)
```
{summary_df.to_csv(index=False)}
```
"""
    filepath.write_text(content)
    print(f"Saved analytics result to {filepath}")
    return str(filepath)
