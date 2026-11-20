# Analysis Results

Dated records of every analysis run.

## Naming convention
`YYYY-MM-DD-short-description.md`

Examples:
- `2026-11-21-quality-screen-bigtech.md`
- `2026-11-21-monte-carlo-nvda.md`
- `2026-11-22-capex-fcf-review.md`

If you run multiple analyses on the same day, add a time or extra identifier:
- `2026-11-21-0945-quality-screen.md`

## What to include in each file
- Date and purpose of the analysis
- Tickers / universe
- Method used (quality screen, Monte Carlo, CapEx/FCF, checklist, etc.)
- Key inputs and assumptions
- Summary results (tables or key numbers)
- Conclusion / next action
- Optional links to raw data, notebooks, or learning notes

## Guidelines
- Commit the summary and key metrics.
- Keep large raw data files out of git (use `data/` or external storage).
- Link related learning notes or checklists when relevant.
- Prefer clear, searchable Markdown.

## Template
See `analysis-results/_template.md` for a starting structure. Copy it and rename for each new analysis.
