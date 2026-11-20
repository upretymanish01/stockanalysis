"""Simple quality and valuation screen inspired by Buffett and Lynch principles.

Uses yfinance info for a lightweight first-pass filter.
Not a full fundamental model — useful for ranking candidates.
"""

from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


def get_quality_metrics(ticker: str) -> Dict:
    """Fetch key quality and valuation metrics for a single ticker."""
    stock = yf.Ticker(ticker)
    info = stock.info

    # Core metrics (many can be None if unavailable)
    metrics = {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("pegRatio"),
        "price_to_book": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),  # already a decimal (e.g. 0.15 = 15%)
        "profit_margin": info.get("profitMargins"),
        "operating_margin": info.get("operatingMargins"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "free_cashflow": info.get("freeCashflow"),
        "operating_cashflow": info.get("operatingCashflow"),
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "dividend_yield": info.get("dividendYield"),
    }
    return metrics


def screen_tickers(
    tickers: List[str],
    min_roe: float = 0.15,
    max_debt_to_equity: float = 1.0,
    max_peg: float = 2.0,
    min_profit_margin: float = 0.10,
) -> pd.DataFrame:
    """
    Screen a list of tickers using simple quality/valuation filters.

    Filters (all optional — set to None to disable):
    - ROE >= min_roe
    - Debt/Equity <= max_debt_to_equity (if available)
    - PEG <= max_peg (if available)
    - Profit margin >= min_profit_margin
    """
    rows = []
    for t in tickers:
        try:
            m = get_quality_metrics(t)
            rows.append(m)
        except Exception as e:
            rows.append({"ticker": t, "error": str(e)})

    df = pd.DataFrame(rows)

    # Apply filters where data exists
    if "roe" in df.columns:
        df = df[df["roe"].isna() | (df["roe"] >= min_roe)]
    if "debt_to_equity" in df.columns and max_debt_to_equity is not None:
        df = df[df["debt_to_equity"].isna() | (df["debt_to_equity"] <= max_debt_to_equity * 100)]  # yfinance often returns D/E as percentage
    if "peg_ratio" in df.columns and max_peg is not None:
        df = df[df["peg_ratio"].isna() | (df["peg_ratio"] <= max_peg)]
    if "profit_margin" in df.columns:
        df = df[df["profit_margin"].isna() | (df["profit_margin"] >= min_profit_margin)]

    # Sort by ROE descending where available
    if "roe" in df.columns:
        df = df.sort_values("roe", ascending=False, na_position="last")

    return df.reset_index(drop=True)


def score_quality(metrics: Dict) -> Optional[float]:
    """
    Simple 0–100 quality score (higher is better).
    Weights are illustrative and can be adjusted.
    """
    score = 0.0
    weights = 0.0

    if metrics.get("roe") is not None:
        score += min(metrics["roe"] * 100, 40) * 0.3   # cap contribution
        weights += 0.3
    if metrics.get("profit_margin") is not None:
        score += min(metrics["profit_margin"] * 100, 30) * 0.25
        weights += 0.25
    if metrics.get("peg_ratio") is not None and metrics["peg_ratio"] > 0:
        peg_score = max(0, 30 - metrics["peg_ratio"] * 10)
        score += peg_score * 0.25
        weights += 0.25
    if metrics.get("debt_to_equity") is not None:
        de = metrics["debt_to_equity"]
        # yfinance debtToEquity is often already *100
        de_ratio = de / 100 if de > 10 else de
        de_score = max(0, 20 - de_ratio * 10)
        score += de_score * 0.2
        weights += 0.2

    if weights == 0:
        return None
    return round(score / weights, 1)
