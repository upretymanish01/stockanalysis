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
│   ├── monte_carlo.py      # GBM path simulation + risk metrics
│   └── utils.py
├── notebooks/
├── tests/
├── data/          # sample data only (large files stay out of git)
└── output/        # generated reports (gitignored)
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
