# stockanalysis

Stock analysis tools and workflows: data fetching, technical/fundamental analysis, screening, and reporting.

Complements the existing Quant series (Quant, QuantBuilder, QuantTrader, Quant-Web).

## Goals
- Fetch market data reliably (Yahoo Finance, alternative sources)
- Perform technical and fundamental analysis
- Build screeners and scoring systems
- Generate clear reports and visualizations
- Keep everything modular, testable, and easy to extend into AI-assisted workflows

## Project Structure
```
stockanalysis/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py
│   ├── analysis.py
│   ├── analytics.py            # retrieve data + multi-period comparison + plots
│   ├── monte_carlo.py          # GBM path simulation + risk metrics
│   ├── quality_screen.py        # Buffett/Lynch-style quality & valuation screen
│   ├── statistical_edge.py      # simple signal / edge tester
│   └── utils.py
├── analysis-results/         # dated records of every analysis run
│   ├── README.md              # naming convention
│   └── _template.md            # copy for new results
├── learning/                 # ongoing notes, concepts, sessions, ideas
│   ├── concepts/               # e.g. top-investors-principles.md
│   ├── sessions/
│   ├── ideas/
│   └── templates/              # stock-checklist.md
├── notebooks/
├── tests/
├── data/                    # sample data only (large files stay out of git)
└── output/                  # generated reports (gitignored)
```

## Analytics — Retrieve & Show Data (Multi-Period)
`src/analytics.py` captures price data once, then lets you compare returns across short and long periods easily.

### Multi-period comparison (recommended for your use case)
Compares 2d / 3d / 4d / 5d, 1w, 1m, 3m, 6m, 1y, 2y in one view.

```python
from src.analytics import multi_period_universe, plot_multi_period_returns, save_analytics_result

tickers = ["AAPL", "MSFT", "GOOGL", "NVDA"]
df = multi_period_universe(tickers)
print(df)

# Visual comparison
plot_multi_period_returns(df, title="Big Tech Multi-Period Returns")

# Save both Markdown + CSV into dated folder
# Includes model name and stocks for your analysis viewer
save_analytics_result(
    df,
    title="bigtech-multi-period-returns",
    model="multi_period_analytics",
    stocks=tickers,
)
```

### Other analytics helpers
```python
from src.analytics import analytics_for_universe, get_price_analytics, plot_price_and_returns

# Single-period universe summary
df = analytics_for_universe(["AAPL", "MSFT"], period="1y")

# Price + cumulative return chart
a = get_price_analytics("AAPL", period="2y")
plot_price_and_returns(a)
```

## Analysis Results (Reusable for Viewer)
Every analysis run automatically creates **two files** in a dated folder:

```
analysis-results/2026-11-21/
├── bigtech-multi-period-returns.md   # human summary + metadata
└── bigtech-multi-period-returns.csv  # clean data for analysis viewer
```

- Markdown contains: model/method used, stocks, summary table, notes
- CSV contains the raw numbers (easy to load in any viewer or future tool)

This local pair is what you can reuse later to reflect results in the analysis viewer.

See `analysis-results/README.md` for the full convention.

## Learning notes
Use the `learning/` folder to keep adding notes as you study.  
See `learning/concepts/top-investors-principles.md` for how Buffett, Lynch, Simons, and Druckenmiller ideas are mapped to tools in this repo.  
Use `learning/templates/stock-checklist.md` as a research template.

## Quality & Valuation Screen
`src/quality_screen.py` provides a lightweight first-pass filter based on ROE, margins, PEG, and debt.

```python
from src.quality_screen import screen_tickers, get_quality_metrics, score_quality

# Screen a list
df = screen_tickers(["AAPL", "MSFT", "GOOGL", "NVDA"], min_roe=0.15, max_peg=2.0)
print(df)

# Single ticker detail
metrics = get_quality_metrics("AAPL")
print(score_quality(metrics))
```

## Statistical Edge Testing
`src/statistical_edge.py` lets you quickly test whether a simple signal has historical edge.

```python
from src.statistical_edge import test_momentum_edge
from src.data_fetcher import fetch_price_history

df = fetch_price_history("AAPL", period="5y")
result = test_momentum_edge(df["Close"], lookback=20, forward_days=5)
print(result)
```

## Monte Carlo Simulations

`src/monte_carlo.py` provides Geometric Brownian Motion (GBM) simulations and risk summaries.

### Quick example
```python
from src.monte_carlo import simulate_stock_paths, summarize_terminal_prices, estimate_params_from_history
from src.data_fetcher import fetch_price_history

# Estimate parameters from history
df = fetch_price_history("AAPL", period="2y")
params = estimate_params_from_history(df["Close"])
print(params)

# Simulate 1-year paths
paths = simulate_stock_paths(
    S0=df["Close"].iloc[-1],
    mu=params["mu"],
    sigma=params["sigma"],
    days=252,
    n_simulations=10_000,
    seed=42,
)

summary = summarize_terminal_prices(paths)
print(summary)
# → mean/median terminal price, percentiles, P(loss), 95% VaR/CVaR, etc.
```

Key functions:
- `geometric_brownian_motion(...)` — raw path array
- `simulate_stock_paths(...)` — DataFrame of paths (easy to plot)
- `summarize_terminal_prices(...)` — mean, percentiles, VaR, CVaR, probability of loss
- `estimate_params_from_history(...)` — annualized μ and σ from a price series

## Backtesting

`src/backtest` tests whether a selection rule would have made money on daily bars.

- A signal may use only data available at the **close of day t**.
- The order fills at the **next session's open**.
- Commission and slippage are set in basis points of traded notional.
- Rebalance with `--rebalance daily`, `weekly`, or `monthly`.
- Strategies in the comparison:
  - **top_n** — the repo's top-5-per-theme composite (risk-adjusted return, pattern score, relative strength vs SPY and the theme ETF), equal-weighted
  - **momentum_equal** / **momentum_vol** — 12-1 momentum, long only if momentum is positive and price is above the 200-day average; equal weight or inverse-volatility sized with a 15% volatility cap
  - **buy_hold_SPY** and **buy_hold_\<theme ETF\>** — the ETF map in `SECTOR_ETF`
- `--oos-start` or `--walk-forward` tunes momentum's lookback on the train window and reports **out-of-sample** rows as the performance record. A run with neither flag is labeled `sample=full` and is not a holdout.

```bash
python -m src.backtest \
  --themes tech,finance,healthcare \
  --start 2018-01-01 \
  --end 2026-11-21 \
  --rebalance monthly \
  --commission-bps 1 \
  --slippage-bps 5 \
  --oos-start 2022-01-01 \
  --output output/backtest
```

Walk-forward (non-overlapping 126-session test windows after 504 in-sample sessions):

```bash
python -m src.backtest \
  --themes tech,finance,healthcare \
  --start 2016-01-01 \
  --rebalance monthly \
  --walk-forward \
  --train-days 504 \
  --test-days 126 \
  --output output/backtest-wf
```

The command writes `summary.csv`, `summary.md`, and `equity_curve.png`. Pass `--prices bars.csv` (columns `date,ticker,open,close,volume`) to run from a file instead of Yahoo Finance.

From Python:

```python
from src.backtest import compare_strategies
from src.backtest.engine import BacktestConfig
from src.backtest.panel import load_yahoo_panel
from src.analytics import FOCUS_CATEGORIES, SECTOR_ETF

tickers = ["SPY", "XLK", *FOCUS_CATEGORIES["tech"]]
panel, skipped = load_yahoo_panel(tickers, start="2018-01-01", end="2026-11-21")
comparison = compare_strategies(
    panel,
    {"tech": FOCUS_CATEGORIES["tech"]},
    config=BacktestConfig(rebalance="monthly", commission_bps=1, slippage_bps=5),
    sector_etfs=SECTOR_ETF,
    oos_start="2022-01-01",
)
print(comparison.summary)
```

**Survivorship bias.** The theme lists are today's listed names. Yahoo Finance does not put delisted, acquired, or bankrupt companies back into those lists, and a renamed ticker may not include the old symbol's history. A backtest on this universe is biased upward versus a point-in-time membership list. The live screener's earnings-growth filter is also omitted unless you pass a point-in-time earnings series, because `yfinance` `.info` is the current snapshot rather than the number known on each decision date. Those limits are repeated in every report the CLI writes.

## Getting Started
```bash
git clone https://github.com/upretymanish01/stockanalysis.git
cd stockanalysis
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run tests
pytest tests/
```

Copy `.env.example` to `.env` and add any API keys if needed.

## Notes
- Prefer small, focused modules.
- Keep notebooks exploratory; move production logic into `src/`.
- No credentials or large data files in the repo.
