"""Leakage, labels, purged splits, ledger immutability, and empty rankings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.panel import panel_from_frames
from src.research.dataset import build_learning_frame
from src.research.errors import classify_outcome
from src.research.features import price_features
from src.research.labels import forward_open_return, label_overlaps_test
from src.research.ledger import LedgerError, PredictionLedger
from src.research.models import evaluate_horizon, rank_latest
from src.research.splits import pre_holdout_train_dates, purged_walk_forward


def _series(n=300, drift=0.001, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-02", periods=n)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), index=index)
    volume = pd.Series(rng.integers(1_000_000, 2_000_000, n), index=index, dtype=float)
    return close, volume


def test_features_at_t_match_a_history_truncated_at_t():
    close, volume = _series()
    bench, _ = _series(seed=1)
    full = price_features(close, volume, bench)
    cut = 250
    truncated = price_features(close.iloc[: cut + 1], volume.iloc[: cut + 1], bench.iloc[: cut + 1])
    row_full = full.iloc[cut]
    row_cut = truncated.iloc[-1]
    assert row_full.index.equals(row_cut.index)
    assert np.allclose(row_full.to_numpy(dtype=float), row_cut.to_numpy(dtype=float), equal_nan=True)


def test_forward_open_label_is_the_next_open_holding_period():
    opens = pd.Series([10, 12, 11, 15, 14, 16, 18], index=pd.bdate_range("2024-01-02", periods=7))
    labels = forward_open_return(opens, horizon=2)
    # Decision on day 0 earns open[3] / open[1] - 1 = 15/12 - 1.
    assert labels.iloc[0] == pytest.approx(15 / 12 - 1)
    assert labels.iloc[1] == pytest.approx(14 / 11 - 1)
    assert pd.isna(labels.iloc[-1])
    assert pd.isna(labels.iloc[-3])


def test_purge_drops_labels_that_reach_the_test_window():
    assert label_overlaps_test(decision_position=16, test_start=20, horizon=5)
    assert not label_overlaps_test(decision_position=13, test_start=20, horizon=5)
    dates = pd.bdate_range("2020-01-02", periods=400)
    folds = list(purged_walk_forward(dates, horizon=21, test_days=40, holdout_days=50, min_train_days=80))
    assert folds
    holdout = dates[-50]
    for train, test in folds:
        assert train.max() < test.min()
        assert test.max() < holdout
        test_pos = int(dates.get_loc(test[0]))
        for stamp in train:
            assert not label_overlaps_test(int(dates.get_loc(stamp)), test_pos, 21)
    trainable = pre_holdout_train_dates(dates, horizon=21, holdout_days=50)
    assert trainable.max() < holdout
    for stamp in trainable:
        assert not label_overlaps_test(int(dates.get_loc(stamp)), int(dates.get_loc(holdout)), 21)


def test_learning_frame_label_matches_the_open_path():
    index = pd.bdate_range("2021-01-04", periods=40)
    opens = np.arange(10, 50, dtype=float)
    closes = opens + 0.5
    panel = panel_from_frames(
        {
            "AAA": pd.DataFrame({"Open": opens, "Close": closes, "Volume": 1e6}, index=index),
            "SPY": pd.DataFrame({"Open": opens, "Close": closes, "Volume": 1e6}, index=index),
        }
    )
    frame = build_learning_frame(panel, ["AAA"], horizon=5, benchmark="SPY", round_trip_cost_bps=12)
    row = frame.iloc[0]
    expected = opens[6] / opens[1] - 1
    assert row["fwd_gross"] == pytest.approx(expected)
    assert row["fwd_net"] == pytest.approx(expected - 0.0012)


def test_prediction_ledger_rejects_rewrites_and_early_outcomes(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = PredictionLedger(path)
    record = {
        "prediction_id": "p1",
        "as_of": "2024-01-02",
        "ticker": "AAA",
        "model_version": "ridge-1",
        "horizon": 21,
        "predicted_return": 0.04,
        "predicted_probability": 0.62,
        "entry_assumption": "next open",
        "exit_assumption": "open 21 sessions later",
    }
    ledger.add(record)
    with pytest.raises(LedgerError):
        changed = dict(record)
        changed["predicted_return"] = 0.99
        ledger.add(changed)
    with pytest.raises(LedgerError):
        ledger.settle("p1", realized_net=0.01, benchmark_return=0.0, sessions_since_decision=10)
    settled = ledger.settle("p1", realized_net=-0.02, benchmark_return=-0.01, sessions_since_decision=21)
    assert settled["status"] == "incorrect"
    assert settled["predicted_return"] == 0.04
    with pytest.raises(LedgerError):
        ledger.settle("p1", realized_net=0.05, benchmark_return=0.0, sessions_since_decision=21)
    reloaded = PredictionLedger(path)
    assert reloaded.predictions["p1"]["predicted_return"] == 0.04
    assert reloaded.predictions["p1"]["status"] == "incorrect"
    assert len(path.read_text().splitlines()) == 2


def test_failure_flags_use_only_observed_returns():
    flags = classify_outcome(
        status="incorrect",
        predicted_return=0.05,
        realized_net=-0.02,
        benchmark_return=-0.03,
        sector_return=-0.04,
    )
    assert flags == ["overestimated_return", "market_decline", "sector_underperformance"]
    assert classify_outcome("pending", 0.1, 0.0, 0.0) == ["pending"]


def test_rank_latest_can_return_nobody():
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "ticker": ["AAA", "BBB"],
            "fwd_net": [0.01, -0.02],
            "mom_252_21": [-0.1, -0.2],
            "dist_sma_200": [-0.05, -0.04],
            "ridge": [-0.01, -0.02],
            "logistic_prob": [0.4, 0.3],
        }
    )
    assert rank_latest(scores, "ridge", min_score=0.0, threshold=None).empty
    assert rank_latest(scores, "logistic", min_score=0.0, threshold=None).empty
    assert rank_latest(scores, "momentum", min_score=0.0, threshold=0.6).empty


def test_horizon_evaluation_does_not_train_through_the_holdout():
    index = pd.bdate_range("2016-01-04", periods=520)
    rng = np.random.default_rng(4)
    frames = {}
    for i, ticker in enumerate(["AAA", "BBB", "CCC", "DDD", "SPY"]):
        shock = rng.normal(0.0004, 0.01, len(index))
        close = 50 * np.exp(np.cumsum(shock + i * 0.0001))
        frames[ticker] = pd.DataFrame(
            {"Open": close, "Close": close, "Volume": 1_500_000.0},
            index=index,
        )
    panel = panel_from_frames(frames)
    frame = build_learning_frame(panel, ["AAA", "BBB", "CCC", "DDD"], horizon=10, round_trip_cost_bps=12)
    report = evaluate_horizon(frame, panel.index, horizon=10, holdout_days=60, test_days=40)
    assert report.selected_model in {"momentum", "ridge", "boosted_tree"}
    assert report.holdout["start"] == str(panel.index[-60].date())
    assert report.holdout["recommendation_count"] >= 0
    assert "not used to choose" in " ".join(report.notes)
    train_dates = pre_holdout_train_dates(panel.index, horizon=10, holdout_days=60)
    assert train_dates.max() < panel.index[-60]
