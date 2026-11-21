"""Analytics module: retrieve data and produce summary analytics + visualizations.

Combines price history, basic stats, quality metrics, and simple plots.
Designed so results can be easily saved into analysis-results/ dated folders.

Includes multi-period comparison so you can capture data once and compare
2/3/4/5 days, 1 week, 1 month, 3 months, 6 months, and years.

Every analysis run:
- Records focus (industry or whole market)
- Returns only the top 5 by composite score
- Includes pattern score, relative strength (vs SPY and sector), volume, Monte Carlo expected return, probability, and 95% VaR
- Suggests conservative position size (1% risk)
- Applies earnings growth filter when available
- Includes current price, 1-year price/return, next 1-month and next 3-month expected returns
- Saves both Markdown and CSV
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

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
    # XYZ is Block, Inc. The SQ ticker was retired on 2025-01-21.
    "finance": ["V", "MA", "PYPL", "XYZ", "AXP"],
    "insurance": ["BRK-B", "PGR", "CB", "MET"],
    "energy": ["XOM", "CVX", "COP"],
    "healthcare": ["JNJ", "UNH", "PFE", "LLY"],
    "consumer staples": ["PG", "KO", "WMT", "COST"],
    "real estate": ["PLD", "AMT", "EQIX"],
}

# Simple sector ETF mapping for relative strength (approximate)
SECTOR_ETF = {
    "chip making": "SMH",
    "ai": "XLK",
    "robotics": "ROBO",
    "renewable": "ICLN",
    "aerospace": "ITA",
    "fashion": "XRT",
    "rockets": "UFO",
    # EDUT (Global X Education ETF) was liquidated in November 2023.
    # The education names in FOCUS_CATEGORIES are GICS consumer discretionary.
    "education": "XLY",
    "services": "XLK",
    "tech": "XLK",
    "banking": "XLF",
    "finance": "XLF",
    "insurance": "KIE",
    "energy": "XLE",
    "healthcare": "XLV",
    "consumer staples": "XLP",
    "real estate": "XLRE",
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


def _relative_strength(ticker: str, benchmark: str = "SPY", period: str = "1y") -> float:
    """Return relative strength vs a benchmark (stock return - benchmark return)."""
    try:
        stock = fetch_price_history(ticker, period=period)
        bench = fetch_price_history(benchmark, period=period)
        if stock.empty or bench.empty:
            missing = [
                name
                for name, frame in ((ticker, stock), (benchmark, bench))
                if frame is None or frame.empty
            ]
            logger.warning(
                "Relative strength unavailable for %s vs %s; missing price data for %s",
                ticker,
                benchmark,
                ", ".join(missing),
            )
            return np.nan
        stock_ret = stock["Close"].iloc[-1] / stock["Close"].iloc[0] - 1
        bench_ret = bench["Close"].iloc[-1] / bench["Close"].iloc[0] - 1
        return float(stock_ret - bench_ret)
    except Exception as exc:
        logger.warning(
            "Relative strength unavailable for %s vs %s: %s",
            ticker,
            benchmark,
            exc,
        )
        return np.nan


def _monte_carlo_for_horizon(close: pd.Series, days: int, n_simulations: int = 5_000) -> Dict:
    """
    Run a quick Monte Carlo for the given horizon and return expected return, probability of gain, and 95% VaR.
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
            "mc_var_95": summary["var_95"],  # negative number = downside
        }
    except Exception as exc:
        logger.warning("Monte Carlo failed for a %s-day horizon: %s", days, exc)
        return {"mc_expected_return": np.nan, "mc_probability_gain": np.nan, "mc_var_95": np.nan}


def _basic_fundamentals(ticker: str) -> Dict:
    """Pull a few useful fundamentals when available."""
    try:
        info = yf.Ticker(ticker).info or {}
        if not info:
            logger.warning("No fundamental data for %s", ticker)
        return {
            "last_quarter_earnings": info.get("earningsQuarterlyGrowth"),
            "operating_expenses": info.get("operatingExpense"),
            "capex": info.get("capitalExpenditures"),
            "total_revenue": info.get("totalRevenue"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception as exc:
        logger.warning("Fundamental fetch failed for %s: %s", ticker, exc)
        return {}


def _suggested_position_pct(
    var_95: Optional[float],
    risk_pct: float = 0.01,
    cap: float = 0.10,
    var_floor: float = 0.01,
    fallback_var: float = 0.05,
) -> float:
    """Fraction of capital to risk `risk_pct` against a 95% VaR return.

    Missing or non-finite VaR used to fall through `nan > 0` and become a 0%
    position. Those cases use `fallback_var` (5% downside) instead.
    """
    var_abs = None
    if var_95 is not None:
        try:
            value = float(var_95)
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and abs(value) > 0:
            var_abs = abs(value)
    if var_abs is None:
        var_abs = float(fallback_var)
    return float(min(risk_pct / max(var_abs, var_floor), cap))


def _frame_has_no_history(frame: pd.DataFrame) -> bool:
    """True when a per-ticker result is only an error row."""
    if frame.empty:
        return True
    return "error" in frame.columns and "timeframe" not in frame.columns


def multi_period_returns(
    ticker: str,
    lookbacks: Optional[List[int]] = None,
    base_period: str = "2y",
    focus: Optional[str] = None,
) -> pd.DataFrame:
    """Capture price data once, then compute total returns over multiple lookback windows."""
    if lookbacks is None:
        lookbacks = [2, 3, 4, 5, 21, 63, 126, 252]

    label_map = {
        2: "2d", 3: "3d", 4: "4d", 5: "5d / 1w",
        21: "1m", 63: "3m", 126: "6m",
        252: "1y",
    }

    # 1m and 3m stay the published forward outlook. Every reported lookback
    # still needs its own VaR so position size is not left at 0%.
    outlook_horizons = {21, 63}
    mc_horizons = set(lookbacks) | outlook_horizons

    df = fetch_price_history(ticker, period=base_period)
    if df is None or df.empty or len(df) < max(lookbacks):
        rows = 0 if df is None else len(df)
        logger.warning(
            "No usable price history for %s (%d rows; need at least %d)",
            ticker,
            rows,
            max(lookbacks),
        )
        return pd.DataFrame([{"ticker": ticker, "error": "Insufficient data"}])

    close = df["Close"]
    latest = close.iloc[-1]
    noise = _noise_score(close)
    pattern = _pattern_score(close)
    rel_spy = _relative_strength(ticker, "SPY", base_period)
    sector_etf = SECTOR_ETF.get(focus.lower() if focus else "", "SPY")
    rel_sector = _relative_strength(ticker, sector_etf, base_period)
    avg_vol = float(df["Volume"].mean())
    fundamentals = _basic_fundamentals(ticker)
    returns = close.pct_change().dropna()

    # 1-year price and return (if enough data)
    one_year_price = np.nan
    one_year_return = np.nan
    if len(close) >= 252:
        one_year_price = float(close.iloc[-252])
        one_year_return = float((latest / one_year_price) - 1)

    mc_cache = {}
    for d in mc_horizons:
        if d < len(close):
            mc_cache[d] = _monte_carlo_for_horizon(close, d)

    # Always compute next 1-month and next 3-month outlook
    next_1m = mc_cache.get(21, {"mc_expected_return": np.nan, "mc_probability_gain": np.nan})
    next_3m = mc_cache.get(63, {"mc_expected_return": np.nan, "mc_probability_gain": np.nan})

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

        mc = mc_cache.get(days, {"mc_expected_return": np.nan, "mc_probability_gain": np.nan, "mc_var_95": np.nan})
        is_outlook = days in outlook_horizons

        # Risk 1% of capital against this horizon's 95% VaR, capped at 10%.
        suggested_position_pct = _suggested_position_pct(mc.get("mc_var_95"))

        rows.append({
            "ticker": ticker,
            "sector": fundamentals.get("sector"),
            "industry": fundamentals.get("industry"),
            "timeframe": label_map.get(days, f"{days}d"),
            "lookback_days": days,
            "current_price": float(latest),
            "one_year_ago_price": one_year_price,
            "one_year_return_pct": round(one_year_return * 100, 2) if not np.isnan(one_year_return) else np.nan,
            "next_1m_expected_return": next_1m["mc_expected_return"],
            "next_1m_probability_gain": next_1m["mc_probability_gain"],
            "next_3m_expected_return": next_3m["mc_expected_return"],
            "next_3m_probability_gain": next_3m["mc_probability_gain"],
            "start_price": float(start_price),
            "end_price": float(latest),
            "actual_return": float(total_ret),
            "actual_return_pct": round(total_ret * 100, 2),
            "risk_adjusted_score": risk_adj,
            "noise_score": noise,
            "pattern_score": pattern,
            "relative_strength_vs_spy": rel_spy,
            "relative_strength_vs_sector": rel_sector,
            "avg_volume": avg_vol,
            "mc_expected_return": mc["mc_expected_return"],
            "mc_probability_gain": mc["mc_probability_gain"],
            "mc_var_95": mc["mc_var_95"],
            "suggested_position_pct": round(suggested_position_pct * 100, 1),
            "last_quarter_earnings": fundamentals.get("last_quarter_earnings"),
            "operating_expenses": fundamentals.get("operating_expenses"),
            "capex": fundamentals.get("capex"),
            "total_revenue": fundamentals.get("total_revenue"),
            "model_name": "monte_carlo_gbm" if is_outlook else None,
            "model_expected_return": mc["mc_expected_return"] if is_outlook else np.nan,
            "model_probability": mc["mc_probability_gain"] if is_outlook else np.nan,
            "prediction_date": datetime.now().strftime("%Y-%m-%d") if is_outlook else None,
            "notes": "Risk 1% of capital; position size capped for safety",
        })

    return pd.DataFrame(rows)


def multi_period_universe(
    tickers: List[str],
    lookbacks: Optional[List[int]] = None,
    base_period: str = "2y",
    focus: Optional[str] = None,
) -> pd.DataFrame:
    """Run multi-period returns for multiple tickers."""
    frames = []
    skipped: List[str] = []
    for t in tickers:
        try:
            frame = multi_period_returns(t, lookbacks, base_period, focus=focus)
        except Exception as exc:
            logger.warning("Skipping %s after data fetch failed: %s", t, exc)
            skipped.append(t)
            frames.append(pd.DataFrame([{"ticker": t, "error": str(exc)}]))
            continue
        if _frame_has_no_history(frame):
            skipped.append(t)
        frames.append(frame)
    if skipped:
        logger.warning(
            "Skipped %d ticker(s) due to data fetch failure: %s",
            len(skipped),
            ", ".join(skipped),
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite score: 40% risk-adjusted, 20% pattern, 20% RS vs SPY, 20% RS vs sector."""
    df = df.copy()
    risk = df["risk_adjusted_score"].fillna(0)
    risk_norm = (risk - risk.min()) / (risk.max() - risk.min() + 1e-9)
    pattern_norm = df["pattern_score"].fillna(0) / 4.0
    rs_spy = df["relative_strength_vs_spy"].fillna(0)
    rs_spy_norm = (rs_spy - rs_spy.min()) / (rs_spy.max() - rs_spy.min() + 1e-9)
    rs_sec = df.get("relative_strength_vs_sector", pd.Series(0, index=df.index)).fillna(0)
    rs_sec_norm = (rs_sec - rs_sec.min()) / (rs_sec.max() - rs_sec.min() + 1e-9)
    df["composite_score"] = 0.40 * risk_norm + 0.20 * pattern_norm + 0.20 * rs_spy_norm + 0.20 * rs_sec_norm
    return df


def top_5_by_timeframe(
    df: pd.DataFrame,
    timeframe: str = "1m",
    rank_by: str = "composite_score",
    max_noise: float = 2.0,
    min_pattern: int = 2,
    min_volume: float = 500_000,
    require_positive_earnings: bool = True,
) -> pd.DataFrame:
    """
    Return only the top 5 tickers for a given timeframe using composite score.

    Filters applied by default:
    - noise_score <= max_noise
    - pattern_score >= min_pattern
    - avg_volume >= min_volume
    - positive last_quarter_earnings when available (if require_positive_earnings)
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
    if require_positive_earnings and "last_quarter_earnings" in subset.columns:
        subset = subset[subset["last_quarter_earnings"].isna() | (subset["last_quarter_earnings"] > 0)]

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
    skipped: List[str] = []
    for t in tickers:
        try:
            a = full_ticker_analytics(t, period)
            if "error" in a:
                logger.warning("Skipping %s: %s", t, a["error"])
                skipped.append(t)
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
            logger.warning("Skipping %s: %s", t, e)
            skipped.append(t)
            rows.append({"ticker": t, "error": str(e)})

    if skipped:
        logger.warning(
            "Skipped %d ticker(s) due to data fetch failure: %s",
            len(skipped),
            ", ".join(skipped),
        )

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
**Ranking:** composite score (risk-adjusted + pattern + RS vs SPY + RS vs sector)  
**Top stocks:** {stocks_str}  
**CSV data:** `{csv_path.name}`  

## Summary Table (Top 5)

{summary_df.to_markdown(index=False) if hasattr(summary_df, 'to_markdown') else summary_df.to_string()}

## Risk Notes
- Position size suggestion is based on risking only 1% of capital using the Monte Carlo 95% VaR.
- Cap at 10% of capital per position.
- This is not a guarantee of returns. Past performance and simulations do not predict the future.
- Always use proper risk management. Never risk more than you can afford to lose.

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
