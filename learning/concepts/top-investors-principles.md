# Top Investors Principles — Operationalized

Extracted repeatable methods from high-performing investors and mapped to actions in this repo.

## Warren Buffett (Berkshire Hathaway)
**Core ideas**
- Own high-quality businesses with durable competitive advantages (moats)
- Strong and consistent free cash flow
- Reasonable valuation relative to intrinsic value
- Long-term ownership; ignore short-term noise
- High return on equity / capital with low debt

**How to use here**
- Run the quality screen in `src/quality_screen.py`
- Fill the stock checklist (`learning/templates/stock-checklist.md`)
- Track FCF and ROE trends over time

## Peter Lynch (Fidelity Magellan)
**Core ideas**
- Growth at a Reasonable Price (GARP) — PEG ratio useful
- Categorize stocks (slow growers, stalwarts, fast growers, cyclicals, turnarounds, asset plays)
- Understand the business simply; invest in what you know
- Look for earnings growth that is sustainable

**How to use here**
- Quality screen includes PEG and growth filters
- Note the category when researching a stock

## Jim Simons (Renaissance / Medallion)
**Core ideas**
- Systematic, data-driven, statistical edge
- Many small, uncorrelated bets rather than few large ones
- Rigorous testing and avoidance of overfitting
- Focus on process over narrative

**How to use here**
- Extend Monte Carlo simulations (`src/monte_carlo.py`) for scenario testing
- Build and backtest simple signals or factors
- Prefer rules that can be coded and tested

## Stanley Druckenmiller
**Core ideas**
- Macro awareness combined with bottom-up research
- Concentrate when conviction is high
- Strong risk management and willingness to change mind quickly
- Liquidity and positioning matter

**How to use here**
- Use checklist to force explicit conviction and risk notes
- Scenario analysis around macro variables (rates, CapEx ROI, etc.)

## Practical integration checklist
- [ ] Run quality screen on candidate
- [ ] Fill stock checklist
- [ ] Estimate parameters and run Monte Carlo for range of outcomes
- [ ] Note what is already priced in vs what could surprise
- [ ] Decide position size based on conviction and risk

## Tags
#buffett #lynch #simons #druckenmiller #process #screening
