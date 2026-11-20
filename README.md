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
│   └── utils.py
├── notebooks/
├── tests/
├── data/          # sample data only (large files stay out of git)
└── output/        # generated reports (gitignored)
```

## Getting Started
```bash
git clone https://github.com/upretymanish01/stockanalysis.git
cd stockanalysis
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add any API keys if needed.

## Notes
- Prefer small, focused modules.
- Keep notebooks exploratory; move production logic into `src/`.
- No credentials or large data files in the repo.
