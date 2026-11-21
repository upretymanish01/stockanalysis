"""Offline backtest tests: next-open fills, costs, buy-and-hold, and walk-forward."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics import top_5_by_timeframe
from src.backtest.compare import WalkForward, compare_strategies, iter_windows, select_momentum_params
from src.backtest.engine import BacktestConfig, decision_dates, run_backtest
from src.backtest.metrics import compute_metrics
from src.backtest.panel import PricePanel, panel_from_frames
from src.backtest.report import SURVIVORSHIP_CAVEAT, write_report
from src.backtest.strategies import MomentumStrategy, TopNThemeStrategy, select_top_n
from src.backtest.__main__ import main
from src.data_fetcher import fetch_price_history


class _LatchLong:
    """Go long after the close has printed `level`, and stay long. History is recorded."""

    warmup = 0
    name = "latch"
    params = {}

    def __init__(self, ticker: str, level: float):
        self.ticker = ticker
        self.level = level
        self.seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def target_weights(self, as_of, history):
        self.seen.append((pd.Timestamp(as_of), pd.Timestamp(history.index.max())))
        if (history.closes[self.ticker] >= self.level).any():
            return pd.Series({self.ticker: 1.0})
        return pd.Series(dtype=float)


def _panel_from_prices(opens: dict[str, list[float]], closes: dict[str, list[float]], volume: float = 1_000_000.0) -> PricePanel:
    n = len(next(iter(closes.values())))
    index = pd.bdate_range("2020-01-02", periods=n)
    frames = {}
    for ticker, close in closes.items():
        frames[ticker] = pd.DataFrame(
            {
                "Open": opens[ticker],
                "High": np.maximum(opens[ticker], close),
                "Low": np.minimum(opens[ticker], close),
                "Close": close,
                "Volume": np.full(n, volume),
            },
            index=index,
        )
    return panel_from_frames(frames)


def _trend_panel(n: int = 260, seed: int = 0) -> PricePanel:
    index = pd.bdate_range("2019-01-02", periods=n)
    rng = np.random.default_rng(seed)
    frames = {}
    specs = {
        "AAA": 0.004,
        "BBB": -0.002,
        "SPY": 0.0002,
        "XLK": 0.0002,
    }
    for ticker, drift in specs.items():
        shock = rng.normal(0, 0.002, n)
        close = 100 * np.exp(np.cumsum(drift + shock))
        open_ = np.concatenate([[close[0]], close[:-1]])
        frames[ticker] = pd.DataFrame(
            {"Open": open_, "Close": close, "Volume": np.full(n, 2_000_000.0)},
            index=index,
        )
    return panel_from_frames(frames)


def test_signal_fills_at_the_next_open_and_does_not_see_the_future():
    # Close jumps through the trigger on day 1. The fill is day 2's open (20),
    # and the book is marked at the last close (40): 40/20 - 1, not the day-1 jump.
    panel = _panel_from_prices(
        opens={"AAA": [10, 10, 20, 22, 30]},
        closes={"AAA": [10, 30, 21, 24, 40]},
    )
    strategy = _LatchLong("AAA", level=30)
    result = run_backtest(panel, strategy, BacktestConfig(rebalance="daily", initial_capital=100))

    assert result.equity.iloc[0] == pytest.approx(100)
    signal_day = panel.index[1]
    assert result.equity.loc[signal_day] == pytest.approx(100)
    assert result.equity.iloc[-1] == pytest.approx(200)
    assert result.metrics["total_return"] == pytest.approx(1.0)
    assert result.total_costs == pytest.approx(0)
    assert result.decision_log
    for row in result.decision_log:
        assert row["history_end"] <= row["decision"]
        assert row["fill"] > row["decision"]
    assert all(history_end <= as_of for as_of, history_end in strategy.seen)
    buy = next(row for row in result.decision_log if row["gross"] > 0)
    assert buy["decision"] == signal_day
    assert buy["fill"] == panel.index[2]


def test_costs_reduce_terminal_equity():
    panel = _panel_from_prices(
        opens={"AAA": [10, 10, 20, 22, 30]},
        closes={"AAA": [10, 30, 21, 24, 40]},
    )
    endings = []
    costs = []
    for bps in (0, 25, 100):
        result = run_backtest(
            panel,
            _LatchLong("AAA", level=30),
            BacktestConfig(rebalance="daily", initial_capital=100, commission_bps=bps, slippage_bps=bps),
        )
        endings.append(result.equity.iloc[-1])
        costs.append(result.total_costs)
    assert endings[0] == pytest.approx(200)
    assert endings[0] > endings[1] > endings[2]
    assert costs[0] == pytest.approx(0)
    assert costs[2] > costs[1] > 0
    # 100 bps slip and 100 bps commission: shares = cash / (open * 1.01 * 1.01)
    expected = 100 / (20 * 1.01 * 1.01) * 40
    assert endings[2] == pytest.approx(expected)


def test_buy_and_hold_matches_close_over_entry_open():
    panel = _panel_from_prices(
        opens={"SPY": [10, 12, 11, 15, 14]},
        closes={"SPY": [11, 13, 12, 16, 18]},
    )
    from src.backtest.strategies import BuyAndHold

    result = run_backtest(panel, BuyAndHold("SPY"), BacktestConfig(rebalance="daily", initial_capital=100))
    simple = 18 / 12 - 1
    assert result.metrics["total_return"] == pytest.approx(simple)
    assert result.equity.iloc[-1] == pytest.approx(100 * 18 / 12)
    assert result.total_costs == pytest.approx(0)
    assert result.turnover == pytest.approx(1.0)
    assert result.metrics["n_rebalances"] == 1


def test_monthly_rebalance_trades_less_often_than_daily():
    index = pd.bdate_range("2021-01-04", periods=260)
    daily = decision_dates(index, "daily", warmup=0)
    monthly = decision_dates(index, "monthly", warmup=0)
    weekly = decision_dates(index, "weekly", warmup=0)
    assert len(monthly) < len(weekly) < len(daily)
    assert 10 <= len(monthly) <= 13


def test_metrics_drawdown_and_hit_rate():
    index = pd.bdate_range("2024-01-02", periods=4)
    equity = pd.Series([100.0, 110.0, 99.0, 108.0], index=index)
    metrics = compute_metrics(equity, turnover=0.5, total_costs=12.5, n_rebalances=1)
    assert metrics["max_drawdown"] == pytest.approx(99 / 110 - 1)
    assert metrics["hit_rate"] == pytest.approx(2 / 3)
    assert metrics["total_costs"] == pytest.approx(12.5)
    returns = equity.pct_change().dropna()
    downside = np.minimum(returns.to_numpy(), 0.0)
    expected_sortino = float(returns.mean() / np.sqrt(np.mean(downside**2)) * np.sqrt(252))
    assert metrics["sortino"] == pytest.approx(expected_sortino)


def test_top_n_selection_matches_the_live_screener_and_ignores_future_bars():
    rows = []
    for ticker, risk, pattern, noise, rs, volume, earnings in [
        ("KEEP", 2.0, 4, 0.8, 0.3, 2_000_000, 0.1),
        ("LOUD", 3.0, 4, 5.0, 0.4, 2_000_000, 0.2),
        ("THIN", 1.5, 4, 0.5, 0.2, 100, 0.3),
        ("WEAK", 0.2, 1, 0.4, -0.1, 2_000_000, 0.4),
        ("LOSS", 1.2, 3, 0.7, 0.1, 2_000_000, -0.2),
        ("ALT", 1.0, 3, 0.9, 0.05, 2_000_000, 0.05),
    ]:
        rows.append(
            {
                "ticker": ticker,
                "timeframe": "1m",
                "risk_adjusted_score": risk,
                "pattern_score": pattern,
                "noise_score": noise,
                "relative_strength_vs_spy": rs,
                "relative_strength_vs_sector": rs,
                "avg_volume": volume,
                "last_quarter_earnings": earnings,
            }
        )
    frame = pd.DataFrame(rows)
    picked = select_top_n(frame, n=5, require_positive_earnings=True)
    live = top_5_by_timeframe(frame, timeframe="1m")
    assert picked["ticker"].tolist() == live["ticker"].tolist()
    assert set(picked["ticker"]) == {"KEEP", "ALT"}

    index = pd.bdate_range("2019-01-02", periods=240)
    up = np.linspace(50, 160, 240)
    down = np.linspace(160, 40, 240)
    flat = np.full(240, 100.0)
    panel = panel_from_frames(
        {
            name: pd.DataFrame(
                {"Open": series, "Close": series, "Volume": np.full(240, 2_000_000.0)},
                index=index,
            )
            for name, series in {"AAA": up, "BBB": down, "SPY": flat, "XLK": flat}.items()
        }
    )
    as_of = panel.index[230]
    strategy = TopNThemeStrategy(
        {"tech": ["AAA", "BBB"]},
        {"tech": "XLK"},
        benchmark="SPY",
        n=5,
        lookback_days=21,
        rs_window=63,
    )
    first = strategy.target_weights(as_of, panel.through(as_of))
    poisoned = panel.copy()
    poisoned.closes.loc[panel.index[231] :, "BBB"] *= 20
    poisoned.opens.loc[panel.index[231] :, "BBB"] *= 20
    second = strategy.target_weights(as_of, poisoned.through(as_of))
    assert panel.through(as_of).index.max() == as_of
    assert first.index.tolist() == second.index.tolist()
    assert "AAA" in first.index
    assert "BBB" not in first.index


def test_earnings_filter_uses_only_prints_known_on_the_decision_date():
    index = pd.bdate_range("2019-01-02", periods=230)
    up = np.linspace(40, 140, 230)
    flat = np.full(230, 100.0)
    panel = panel_from_frames(
        {
            name: pd.DataFrame(
                {"Open": series, "Close": series, "Volume": np.full(230, 2_000_000.0)},
                index=index,
            )
            for name, series in {"AAA": up, "BBB": up, "SPY": flat, "XLK": flat}.items()
        }
    )
    as_of = panel.index[220]
    future = as_of + pd.Timedelta(days=5)
    past = as_of - pd.Timedelta(days=20)
    earnings = {
        "AAA": pd.Series({past: 0.25, future: -0.80}),
        "BBB": pd.Series({past: -0.30}),
    }
    strategy = TopNThemeStrategy(
        {"tech": ["AAA", "BBB"]},
        {"tech": "XLK"},
        n=5,
        lookback_days=21,
        rs_window=63,
        earnings=earnings,
    )
    weights = strategy.target_weights(as_of, panel.through(as_of))
    assert "AAA" in weights.index
    assert "BBB" not in weights.index


def test_momentum_params_ignore_prices_after_the_train_end():
    panel = _trend_panel(n=420, seed=2)
    train_end = panel.index[300]
    kwargs = {"trend_window": 30, "top_n": 1, "vol_lookback": 15, "target_vol": None}
    grid = ((40, 5), (80, 10))
    choice, rows = select_momentum_params(
        panel,
        ["AAA", "BBB"],
        grid,
        BacktestConfig(rebalance="monthly"),
        trade_start=panel.index[0],
        train_end=train_end,
        strategy_kwargs=kwargs,
    )
    poisoned = panel.copy()
    poisoned.closes.iloc[301:] *= 30
    poisoned.opens.iloc[301:] *= 30
    choice_poisoned, _ = select_momentum_params(
        poisoned,
        ["AAA", "BBB"],
        grid,
        BacktestConfig(rebalance="monthly"),
        trade_start=panel.index[0],
        train_end=train_end,
        strategy_kwargs=kwargs,
    )
    assert choice == choice_poisoned
    assert all(pd.Timestamp(row["train_end"]) == pd.Timestamp(train_end).normalize() for row in rows)


def test_walk_forward_windows_do_not_overlap_and_oos_history_stops_at_the_decision():
    index = pd.bdate_range("2018-01-02", periods=800)
    windows = iter_windows(index, WalkForward(train_days=504, test_days=126, step_days=126))
    assert windows
    for train, test in windows:
        assert train[-1] < test[0]
        assert len(train) == 504
        assert len(test) == 126
    assert windows[0][1][-1] < windows[1][1][0]
    partial = iter_windows(index, WalkForward(train_days=504, test_days=126, step_days=126), include_partial=True)
    assert len(partial) == len(windows) + 1
    assert partial[-1][0][-1] < partial[-1][1][0]
    assert len(partial[-1][1]) < 126
    with pytest.raises(ValueError):
        WalkForward(train_days=100, test_days=20, step_days=10)

    panel = _trend_panel(n=800, seed=3)
    comparison = compare_strategies(
        panel,
        {"tech": ["AAA", "BBB"]},
        config=BacktestConfig(rebalance="monthly", commission_bps=1, slippage_bps=5),
        sector_etfs={"tech": "XLK"},
        walk_forward=WalkForward(train_days=504, test_days=126, step_days=126),
        momentum_grid=((126, 21), (63, 10)),
        momentum_sizing="equal",
    )
    assert set(comparison.summary["sample"]) == {"out_of_sample"}
    assert {"top_n", "momentum_equal", "buy_hold_SPY", "buy_hold_XLK"} <= set(comparison.summary["strategy"])
    for result in comparison.results:
        assert result.decision_log
        assert all(row["history_end"] <= row["decision"] for row in result.decision_log)
    assert comparison.tuning
    assert all(pd.Timestamp(row["train_end"]) < pd.Timestamp(row["test_start"]) for row in comparison.tuning)


def test_report_and_cli_run_offline(tmp_path):
    panel = _trend_panel(n=400, seed=4)
    comparison = compare_strategies(
        panel,
        {"tech": ["AAA", "BBB"]},
        config=BacktestConfig(rebalance="monthly"),
        sector_etfs={"tech": "XLK"},
    )
    paths = write_report(comparison, tmp_path / "report")
    text = (tmp_path / "report" / "summary.md").read_text()
    assert "Survivorship bias" in text
    assert SURVIVORSHIP_CAVEAT.split(":")[0] in text
    assert "sample=full" in text
    chart = tmp_path / "report" / "equity_curve.png"
    assert chart.stat().st_size > 1000
    assert paths["summary_csv"].endswith("summary.csv")

    rows = []
    for ticker in panel.tickers:
        for date, open_, close, volume in zip(
            panel.index, panel.opens[ticker], panel.closes[ticker], panel.volumes[ticker]
        ):
            rows.append({"date": date, "ticker": ticker, "open": open_, "close": close, "volume": volume})
    csv_path = tmp_path / "bars.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    # The loader maps theme keys to the repo universe. Build a file that uses those symbols
    # by renaming AAA/BBB/XLK onto the tech book for the CLI smoke test.
    renamed = pd.DataFrame(rows).replace({"AAA": "AAPL", "BBB": "MSFT", "XLK": "XLK", "SPY": "SPY"})
    renamed.to_csv(csv_path, index=False)
    out = tmp_path / "cli"
    code = main(
        [
            "--prices",
            str(csv_path),
            "--themes",
            "tech",
            "--start",
            "2019-01-01",
            "--end",
            "2021-12-31",
            "--rebalance",
            "monthly",
            "--commission-bps",
            "1",
            "--slippage-bps",
            "5",
            "--momentum-sizing",
            "equal",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "summary.md").exists()
    assert "Survivorship bias" in (out / "summary.md").read_text()
    with pytest.raises(SystemExit):
        main(["--prices", str(csv_path), "--themes", "not-a-theme", "--output", str(out)])


def test_price_history_uses_start_end_instead_of_period():
    with patch("src.data_fetcher.yf.Ticker") as ticker_cls:
        ticker_cls.return_value.history.return_value = pd.DataFrame({"Close": [1.0]})
        fetch_price_history("SPY", start="2020-01-01", end="2020-06-01")
    kwargs = ticker_cls.return_value.history.call_args.kwargs
    assert kwargs["start"] == "2020-01-01"
    assert kwargs["end"] == "2020-06-01"
    assert "period" not in kwargs


def test_momentum_signal_uses_skipped_window_only():
    index = pd.bdate_range("2022-01-03", periods=80)
    close = pd.Series(np.linspace(100, 180, 80), index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    flat = pd.Series(100.0, index=index)
    panel = panel_from_frames(
        {
            "AAA": pd.DataFrame({"Open": open_, "Close": close, "Volume": 1e6}, index=index),
            "BBB": pd.DataFrame({"Open": flat, "Close": flat, "Volume": 1e6}, index=index),
        }
    )
    strategy = MomentumStrategy(["AAA", "BBB"], lookback=40, skip=5, top_n=1, trend_window=20, sizing="equal")
    as_of = index[-1]
    weights = strategy.target_weights(as_of, panel.through(as_of))
    past = close.iloc[-1 - 5] / close.iloc[-1 - 40] - 1
    recent_only = close.iloc[-1] / close.iloc[-1 - 5] - 1
    assert past > 0
    assert weights.index.tolist() == ["AAA"]
    # The most recent `skip` sessions are not the momentum window.
    assert past != pytest.approx(recent_only)


def test_missing_close_keeps_the_last_price_instead_of_dropping_the_position():
    # Fill at day-2 open 20 after the day-1 close hits 30. Day 3 has no print.
    # Shares = 100/20 = 5. Last close is 21, so equity stays 105 rather than
    # collapsing to leftover cash (zero).
    panel = _panel_from_prices(
        opens={"AAA": [10, 10, 20, np.nan, 22]},
        closes={"AAA": [10, 30, 21, np.nan, 25]},
    )
    result = run_backtest(
        panel,
        _LatchLong("AAA", level=30),
        BacktestConfig(rebalance="daily", initial_capital=100, delist_after=10),
    )
    hole = panel.index[3]
    assert result.equity.loc[hole] == pytest.approx(105)
    assert result.equity.iloc[-1] == pytest.approx(125)
    assert result.metrics["n_delistings"] == 0
    assert result.metrics["n_stale_marks"] == 1
    assert any(event["action"] == "stale_mark" and event["ticker"] == "AAA" for event in result.valuation_events)


def test_extended_missing_prices_are_written_off_as_a_delisting():
    panel = _panel_from_prices(
        opens={"AAA": [10, 10, 20, np.nan, np.nan, np.nan]},
        closes={"AAA": [10, 30, 21, np.nan, np.nan, np.nan]},
    )
    result = run_backtest(
        panel,
        _LatchLong("AAA", level=30),
        BacktestConfig(rebalance="daily", initial_capital=100, delist_after=2),
    )
    # Two missing sessions write the holding off at the last close (21).
    assert result.equity.iloc[-1] == pytest.approx(0)
    delistings = [event for event in result.valuation_events if event["action"] == "delisted"]
    assert len(delistings) == 1
    assert delistings[0]["loss"] == pytest.approx(105)
    assert result.metrics["n_delistings"] == 1


class _Rotate:
    warmup = 0
    name = "rotate"
    params = {}

    def __init__(self):
        self.calls = 0

    def target_weights(self, as_of, history):
        self.calls += 1
        if self.calls <= 2:
            return pd.Series({"AAA": 1.0})
        return pd.Series({"BBB": 1.0})


def test_suspended_holding_is_not_resold_or_double_counted():
    panel = _panel_from_prices(
        opens={"AAA": [10, 10, np.nan, np.nan], "BBB": [10, 10, 10, 10]},
        closes={"AAA": [10, 10, np.nan, np.nan], "BBB": [10, 10, 10, 12]},
    )
    result = run_backtest(
        panel,
        _Rotate(),
        BacktestConfig(rebalance="daily", initial_capital=100, delist_after=10),
    )
    # Bought 10 AAA shares at 10. The suspension blocks the sale, so the book
    # stays in AAA and is not used as cash to buy BBB.
    assert result.equity.iloc[-1] == pytest.approx(100)
    assert all(event["ticker"] != "BBB" or event["action"] != "stale_mark" for event in result.valuation_events)
