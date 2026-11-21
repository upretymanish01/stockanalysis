# Research status

This file records what is implemented, one real-data run, and what is not built.
It does not claim a 90% hit rate or a tradable edge.

## Architecture

The repository already had Yahoo downloads, a composite top-5 screener, a
next-open backtest, and walk-forward momentum. This branch adds safer
valuation when prices are missing, and a separate research path that labels
21-, 42-, and 63-session open-to-open returns without using those labels as
features.

There is no order router. `python -m src.research` only prints a report.

## Audit

`_mark()` used to ignore a holding when that session had no price. The position
disappeared from equity for the day and, if a rebalance ran, its value could be
treated as if it were not capital. The engine now:

- marks a missing print at the last positive price and logs `stale_mark`
- writes the shares off at zero after `delist_after` sessions (default 10) with
  neither an open nor a close, and logs `delisted` with the loss
- refuses to sell a name that has no open, and does not spend that locked value
  on names that are trading

On the downloaded panel below, every name had a price every session, so this
run recorded zero stale marks and zero delistings. The synthetic tests cover
the hole and the write-off.

Other limits that are still true:

- Theme lists are today's survivors. Delisted names are absent.
- Yahoo returned XYZ history back to 2016. That is the vendor series, not a
  point-in-time membership file, and it may stitch the old SQ history.
- Adjusted OHLC follows Yahoo's corporate-action adjustment. There is no
  separate dividend or split ledger.
- `quality_screen` and the live screener still read the current `.info`
  snapshot. The research model does not.
- Portfolio `hit_rate` is the fraction of up days. Recommendation precision is
  a different number and is labeled `precision_profitable`.
- Costs are a flat 1 bp commission plus 5 bp slippage. There is no spread model
  and no capacity limit.
- `bench_*` features are identical for every name on a given day, so a
  cross-sectional rank correlation for them is undefined.

## What was run

Command:

```bash
python -m src.research \
  --themes tech,finance,healthcare \
  --start 2016-01-01 \
  --end 2026-11-21 \
  --holdout-days 126
```

Data: yfinance adjusted daily OHLC, requested 2016-01-01 through 2026-11-21.
Last session in the file was 2026-11-20. No ticker was skipped. Seventeen
symbols (the three themes, SPY, XLK, XLF, XLV). Missing opens: 0. Missing
closes: 0.

Strategy baseline: monthly rebalance, $100,000, 1 bp commission, 5 bp slippage,
walk-forward of 504 train sessions and 126 test sessions. Momentum's lookback
was chosen only on the preceding train window. Reported path: 2018-01-31
through 2026-11-20, sample `out_of_sample`.

| strategy | CAGR | Sharpe | max drawdown | ann. turnover | costs | ending equity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top_n | 10.63% | 0.59 | -35.1% | 8.38x | $7,717 | $242,837 |
| momentum_equal | 18.36% | 0.83 | -33.0% | 6.30x | $8,471 | $439,766 |
| momentum_vol | 11.81% | 0.72 | -26.6% | 5.83x | $5,253 | $266,690 |
| buy_hold_SPY | 12.44% | 0.71 | -33.7% | 0.11x | $60 | $280,082 |
| buy_hold_XLK | 19.39% | 0.79 | -33.6% | 0.11x | $60 | $474,340 |
| buy_hold_XLF | 9.18% | 0.50 | -42.9% | 0.11x | $60 | $216,316 |
| buy_hold_XLV | 10.16% | 0.64 | -28.4% | 0.11x | $60 | $233,969 |

Sharpe uses a zero risk-free rate and 252 trading days. Over this survivor
universe, equal-weight momentum beat SPY and the top-5 composite did not.
XLK beat every active rule. That is not a claim of alpha: the stock list is
small, partly technology, and missing every name that no longer trades.

The last 126 sessions (2026-05-26 through 2026-11-20) are a short slice of that
same walk-forward equity curve. Total return, not the annualized rate:

| strategy | total return |
| --- | ---: |
| top_n | +0.4% |
| momentum_equal | +16.5% |
| momentum_vol | +16.5% |
| buy_hold_SPY | -7.6% |
| buy_hold_XLK | -19.3% |
| buy_hold_XLF | +6.0% |
| buy_hold_XLV | +20.3% |

Annualized Sharpe on this slice is not comparable to the eight-year figure.

## 21 / 42 / 63 session models

Features are trailing momentum (1, 3, 6, and 12-minus-1 months), distance from
the 50- and 200-day averages, 21- and 63-day volatility, 63-day drawdown,
21-day volume versus its average, and 63-day strength versus SPY. The label is
`open[t+1+horizon] / open[t+1] - 1`, minus 12 bp for a round trip (the same
6 bp per side as the backtest). Rows whose exit open is not in the file are
unlabeled.

Model choice uses only dates before 2026-05-26. Overlapping labels that reach
the test window are purged. Seventeen purged folds, 27,846 labeled rows each
horizon. The selected score is the one with the highest mean daily Spearman
correlation on those folds. Ties would prefer the simpler score. The holdout
starts 2026-05-26 and was not used for that choice or for the logistic
threshold.

Validation mean daily IC:

| horizon | momentum | ridge | boosted tree |
| --- | ---: | ---: | ---: |
| 21 | 0.030 | 0.009 | -0.028 |
| 42 | 0.051 | 0.047 | -0.001 |
| 63 | 0.067 | 0.051 | -0.016 |

Momentum was selected at every horizon. The tree model was not.

Development-sample feature IC (pre-holdout, labels still overlap each other so
the days are not independent draws) was small. The largest positive values were
63-day volatility (0.060 / 0.070 / 0.087 at 21 / 42 / 63 sessions) and distance
above the 200-day average (0.032 / 0.054 / 0.074). One-month momentum was about
zero. On the holdout those same features' rank correlations were negative,
except 21-day volume ratio at the 63-session horizon (+0.03). Market-wide
features do not vary across names inside a day, so their cross-sectional IC is
undefined.

Holdout recommendation precision is the fraction of picks with a positive net
holding-period return. Momentum recommends at most one name per day, and only
when 12-1 momentum is positive and the close is above the 200-day average.
The logistic threshold 0.70 was the validation grid point with the highest
precision among thresholds that produced at least 30 validation picks
(validation precision about 0.64 to 0.65). That validation precision is not
the holdout result.

| horizon | momentum picks | momentum precision | momentum mean net | all-names base rate | logistic picks | logistic precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 21 | 104 | 50.0% | +0.7% | 64.4% | 4 | 0% |
| 42 | 83 | 44.6% | +1.0% | 68.4% | 4 | 0% |
| 63 | 62 | 17.7% | -4.8% | 70.2% | 51 | 51.0% |

Holdout momentum rank IC was -0.18, -0.34, and -0.41. The 90% precision target
was not met. Picks were less often profitable than the average name in the
same holdout. `rank_latest` returns an empty table when no name clears the bar.

## References used for the design

- Jegadeesh and Titman, "Returns to Buying Winners and Selling Losers," Journal
  of Finance, 1993. Past medium-term winners tend to keep outperforming over
  the next several months. That is why 3-, 6-, and 12-month momentum are
  features, and why the baseline strategy is allowed to hold cash when
  momentum is negative.
- Novy-Marx, "Is Momentum Really Momentum?," Journal of Financial Economics,
  2012. Skipping the most recent month is the 12-1 feature and the default
  momentum strategy.
- Lopez de Prado, Advances in Financial Machine Learning, Wiley, 2018. Training
  labels that overlap the test window are removed. That is the purge in
  `src/research/splits.py`.

Fundamental ratios were not added. A current Yahoo snapshot would use numbers
that were not known on the decision date, and this project does not have
filing dates.

## Prediction ledger

`PredictionLedger` appends a forecast line and, later, a settlement line.
Loading the file replays both. A second write cannot change the forecast, and
settlement is rejected until `horizon` sessions have elapsed. Failed outcomes
are tagged only from the numbers: forecast above the realized net return,
negative benchmark return, negative sector return. Those tags can stack. They
are not a narrative of why the stock moved.

No live prediction file is checked in. The ledger is the mechanism; it has not
been run as a production journal.

## Not done

- Scheduled retraining, a promotion gate, and rollback of a model artifact.
- Point-in-time fundamentals and a historical membership list.
- Confidence intervals, performance by regime, and liquidity caps beyond the
  volume feature.
- GitHub Actions, secret handling, and any broker or paper-trading connection.
- Automatic real-money orders. Those are out of scope on purpose.

## Tests

`python3 -m pytest tests/ -q` on this branch: 45 passed, 0 failed, 0 skipped.
Synthetic tests do not show that the book makes money.
