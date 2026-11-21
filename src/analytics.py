"""Analytics module: retrieve data and produce summary analytics + visualizations.

Combines price history, basic stats, quality metrics, and simple plots.
Designed so results can be easily saved into analysis-results/ dated folders.

Includes multi-period comparison so you can capture data once and compare
2/3/4/5 days, 1 week, 1 month, 3 months, 6 months, and years.

Every analysis run:
- Records focus (industry or whole market)
- Returns only the top 5 by composite score (more accurate)
- Includes pattern score, relative strength vs SPY, volume, and Monte Carlo expected return + probability
- Saves both Markdown and CSV
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from .data_fetcher import fetch_price_history
from .monte_carlo import estimate_params_from_history, simulate_stock_paths, summarize_terminal_prices
from .quality_screen import get_quality_metrics

# Expanded focus categories → example tickers
FOCUS_CATEGORIES = {
    "chip making": ["NVDA", "TSM", "AVGO", "AMD", "ASML"],
    "ai": ["NVDA", "MSFT", "GOOGL", "META", "PLTR"],
    "robotics": ["ISRG", "PATH", "SYM", "TER"],
    "renewable": ["ENPH", "FSLR", "NEE", "PLUG"],
    "aerospace": ["BA", "RTX", "LMT", "NOC"],
    "fashion": ["NKE", "LULU", "PVH", "TPR"],
    "rockets": ["RKLB", "ASTS", "LUNR"],
    "education": ["DUOL", "CHGG", "STRA"],
    "services": ["CRM", "NOW", "ADBE", "INTU"],
    "tech": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "banking": ["JPM", "BAC", "WFC", "GS", "MS"],
    "finance": ["V", "MA", "PYPL", "SQ", "AXP"],
    "insurance": ["BRK-B", "PGR", "CB", "MET"],
    "energy": ["XOM", "CVX", "COP"],
    "healthcare": ["JNJ", "UNH", "PFE", "LLY"],
    "consumer staples": ["PG", "KO", "WMT", "COST"],
    "real estate": ["PLD", "AMT", "EQIX"],
}


def get_price_analytics(ticker: str, period: str = "1y") -> Dict:
    """Fetch price history and compute key analytics."""
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


def _noise_score(close: pd.Series, window: int = 20) -> float:
    """Simple noise proxy: recent volatility relative to longer-term volatility."""
    returns = close.pct_change().dropna()
    if len(returns) < window * 2:
        return np.nan
    recent_vol = returns.tail(window).std()
    longer_vol = returns.std()
    if longer_vol == 0:
        return np.nan
    return float(recent_vol / longer_vol)


def _pattern_score(close: pd.Series) -> int:
    """Simple pattern score 0-4 based on moving averages and higher low."""
    if len(close) < 200:
        return 0
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    latest = close.iloc[-1]
    score = 0
    if latest > ma50:
        score += 1
    if ma50 > ma200:
        score += 1
    recent_low = close.tail(20).min()
    prior_low = close.iloc[-40:-20].min() if len(close) >= 40 else recent_low
    if recent_low > prior_low:
        score += 1
    if latest > close.tail(20).mean():
        score += 1
    return score


def _relative_strength(ticker: str, period: str = "1y") -> float:
    """Return relative strength vs SPY over the period (stock return - SPY return)."""
    try:
        stock = fetch_price_history(ticker, period=period)
        spy = fetch_price_history("SPY", period=period)
        if stock.empty or spy.empty:
            return np.nan
        stock_ret = stock["Close"].iloc[-1] / stock["Close"].iloc[0] - 1
        spy_ret = spy["Close"].iloc[-1] / spy["Close"].iloc[0] - 1
        return float(stock_ret - spy_ret)
    except Exception:
        return np.nan


def _monte_carlo_for_horizon(close: pd.Series, days: int, n_simulations: int = 5_000) -> Dict:
    """
    Run a quick Monte Carlo for the given horizon and return expected return + probability of gain.
    """
    try:
        params = estimate_params_from_history(close)
        S0 = float(close.iloc[-1])
        paths = simulate_stock_paths(
            S0=S0,
            mu=params["mu"],
            sigma=params["sigma"],
            days=days,
            n_simulations=n_simulations,
            seed=42,
        )
        summary = summarize_terminal_prices(paths)
        return {
            "mc_expected_return": summary["mean_return"],
            "mc_probability_gain": summary["prob_gain"],
        }
    except Exception:
        return {"mc_expected_return": np.nan, "mc_probability_gain": np.nan}


def _basic_fundamentals(ticker: str) -> Dict:
    """Pull a few useful fundamentals when available."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "last_quarter_earnings": info.get("earningsQuarterlyGrowth"),
            "operating_expenses": info.get("operatingExpense"),
            "capex": info.get("capitalExpenditures"),
            "total_revenue": info.get("totalRevenue"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception:
        return {}


def multi_period_returns(
    ticker: str,
    lookbacks: Optional[List[int]] = None,
    base_period: str = "2y",
) -> pd.DataFrame:
    """Capture price data once, then compute total returns over multiple lookback windows."""
    if lookbacks is None:
        lookbacks = [2, 3, 4, 5, 21, 63, 126, 252]

    label_map = {
        2: "2d", 3: "3d", 4: "4d", 5: "5d / 1w",
        21: "1m", 63: "3m", 126: "6m",
        252: "1y",
    }

    # Horizons where we run Monte Carlo
    mc_horizons = {21, 63}

    df = fetch_price_history(ticker, period=base_period)
    if df.empty or len(df) < max(lookbacks):
        return pd.DataFrame([{"ticker": ticker, "error": "Insufficient data"}])

    close = df["Close"]
    latest = close.iloc[-1]
    noise = _noise_score(close)
    pattern = _pattern_score(close)
    rel_strength = _relative_strength(ticker, base_period)
    avg_vol = float(df["Volume"].mean())
    fundamentals = _basic_fundamentals(ticker)
    returns = close.pct_change().dropna()

    # Pre-compute Monte Carlo for 1m and 3m
    mc_cache = {}
    for d in mc_horizons:
        if d < len(close):
            mc_cache[d] = _monte_carlo_for_horizon(close, d)

    rows = []
    for days in lookbacks:
        if days >= len(close):
            continue
        start_idx = max(0, len(close) - days - 1)
        start_price = close.iloc[start_idx]
        total_ret = (latest / start_price) - 1

        window_returns = returns.tail(days) if days < len(returns) else returns
        vol = window_returns.std()
        risk_adj = float(total_ret / vol) if vol and vol > 0 else np.nan

        mc = mc_cache.get(days, {"mc_expected_return": np.nan, "mc_probability_gain": np.nan})

        rows.append({
            "ticker": ticker,
            "sector": fundamentals.get("sector"),
            "industry": fundamentals.get("industry"),
            "timeframe": label_map.get(days, f"{days}d"),
            "lookback_days": days,
            "start_price": float(start_price),
            "end_price": float(latest),
            "actual_return": float(total_ret),
            "actual_return_pct": round(total_ret * 100, 2),
            "risk_adjusted_score": risk_adj,
            "noise_score": noise,
            "pattern_score": pattern,
            "relative_strength_vs_spy": rel_strength,
            "avg_volume": avg_vol,
            "mc_expected_return": mc["mc_expected_return"],
            "mc_probability_gain": mc["mc_probability_gain"],
            "last_quarter_earnings": fundamentals.get("last_quarter_earnings"),
            "operating_expenses": fundamentals.get("operating_expenses"),
            "capex": fundamentals.get("capex"),
            "total_revenue": fundamentals.get("total_revenue"),
            "model_name": "monte_carlo_gbm" if days in mc_horizons else None,
            "model_expected_return": mc["mc_expected_return"],
            "model_probability": mc["mc_probability_gain"],
            "prediction_date": datetime.now().strftime("%Y-%m-%d") if days in mc_horizons else None,
            "notes": None,
        })

    return pd.DataFrame(rows)


def multi_period_universe(
    tickers: List[str],
    lookbacks: Optional[List[int]] = None,
    base_period: str = "2y",
) -> pd.DataFrame:
    """Run multi-period returns for multiple tickers."""
    frames = []
    for t in tickers:
        frames.append(multi_period_returns(t, lookbacks, base_period))
    return pd.concat(frames, ignore_index=True)


def _add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite score: 50% risk-adjusted, 25% pattern, 25% relative strength."""
    df = df.copy()
    risk = df["risk_adjusted_score"].fillna(0)
    risk_norm = (risk - risk.min()) / (risk.max() - risk.min() + 1e-9)
    pattern_norm = df["pattern_score"].fillna(0) / 4.0
    rs = df["relative_strength_vs_spy"].fillna(0)
    rs_norm = (rs - rs.min()) / (rs.max() - rs.min() + 1e-9)
    df["composite_score"] = 0.50 * risk_norm + 0.25 * pattern_norm + 0.25 * rs_norm
    return df


def top_5_by_timeframe(
    df: pd.DataFrame,
    timeframe: str = "1m",
    rank_by: str = "composite_score",
    max_noise: float = 2.0,
    min_pattern: int = 2,
    min_volume: float = 500_000,
) -> pd.DataFrame:
    """
    Return only the top 5 tickers for a given timeframe using composite score.

    Filters applied by default:
    - noise_score <= max_noise
    - pattern_score >= min_pattern
    - avg_volume >= min_volume
    """
    if df.empty or "timeframe" not in df.columns:
        return df

    subset = df[df["timeframe"] == timeframe].copy()
    if subset.empty:
        return subset

    if "noise_score" in subset.columns:
        subset = subset[subset["noise_score"].isna() | (subset["noise_score"] <= max_noise)]
    if "pattern_score" in subset.columns:
        subset = subset[subset["pattern_score"].fillna(0) >= min_pattern]
    if "avg_volume" in subset.columns:
        subset = subset[subset["avg_volume"].fillna(0) >= min_volume]

    subset = _add_composite_score(subset)

    if rank_by not in subset.columns:
        rank_by = "composite_score"

    subset = subset.sort_values(rank_by, ascending=False, na_position="last")
    return subset.head(5).reset_index(drop=True)


def get_tickers_for_focus(focus: str) -> List[str]:
    """Return example tickers for a focus category."""
    key = focus.lower().strip()
    return FOCUS_CATEGORIES.get(key, [])


def attach_model_predictions(
    df: pd.DataFrame,
    model_name: str,
    predictions: Dict[str, Dict],
    prediction_date: Optional[str] = None,
) -> pd.DataFrame:
    """Attach model predictions to an existing multi-period DataFrame."""
    if prediction_date is None:
        prediction_date = datetime.now().strftime("%Y-%m-%d")

    df = df.copy()
    for ticker, pred in predictions.items():
        mask = df["ticker"] == ticker
        df.loc[mask, "model_name"] = model_name
        df.loc[mask, "model_expected_return"] = pred.get("expected_return")
        df.loc[mask, "model_probability"] = pred.get("probability")
        df.loc[mask, "prediction_date"] = prediction_date
        df.loc[mask, "notes"] = pred.get("notes")
    return df


def plot_multi_period_returns(
    df: pd.DataFrame,
    title: str = "Multi-Period Returns Comparison",
    save_path: Optional[str] = None,
):
    """Bar chart of actual returns across periods."""
    if df.empty or "error" in df.columns:
        print("No valid data to plot")
        return

    value_col = "actual_return_pct" if "actual_return_pct" in df.columns else "total_return_pct"
    pivot = df.pivot(index="timeframe" if "timeframe" in df.columns else "period_label",
                     columns="ticker", values=value_col)

    preferred_order = ["2d", "3d", "4d", "5d / 1w", "1m", "3m", "6m", "1y"]
    existing = [p for p in preferred_order if p in pivot.index]
    if existing:
        pivot = pivot.reindex(existing)

    ax = pivot.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_title(title)
    ax.set_ylabel("Actual Return (%)")
    ax.set_xlabel("Timeframe")
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
    """Combined price analytics + quality metrics for one ticker."""
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
    """Run full analytics on a list of tickers and return a summary DataFrame."""
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
    focus: str = "whole market",
    model: str = "multi_period_analytics",
    stocks: Optional[List[str]] = None,
    date_str: Optional[str] = None,
    notes: str = "",
) -> Dict[str, str]:
    """
    Save both Markdown summary and CSV data into a dated folder under analysis-results/.

    focus: "whole market" or specific industry name
    Always intended to be used with top-5 results.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    folder = Path("analysis-results") / date_str
    folder.mkdir(parents=True, exist_ok=True)

    safe_title = title.lower().replace(" ", "-").replace("/", "-")
    md_path = folder / f"{safe_title}.md"
    csv_path = folder / f"{safe_title}.csv"

    if stocks is None and "ticker" in summary_df.columns:
        stocks = summary_df["ticker"].unique().tolist()
    stocks_str = ", ".join(stocks) if stocks else "N/A"

    summary_df.to_csv(csv_path, index=False)

    content = f"""# Analysis Result — {date_str}

**Title:** {title}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
**Focus:** {focus}  
**Model / Method:** {model}  
**Ranking:** composite score (risk-adjusted + pattern + relative strength)  
**Top stocks:** {stocks_str}  
**CSV data:** `{csv_path.name}`  

## Summary Table (Top 5)

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
