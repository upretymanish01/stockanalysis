"""Walk-forward comparison of a momentum score, ridge, a tree model, and logistic.

The model that is reported as selected is chosen only on pre-holdout folds,
by mean daily Spearman rank correlation of the score with subsequent net
return. The holdout is scored once, after that choice, and is not used to
change it. A logistic threshold is frozen the same way. If no validation
threshold produces enough recommendations, the probability rule recommends
nobody.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from .features import FEATURE_COLUMNS
from .splits import holdout_start, pre_holdout_train_dates, purged_walk_forward

MIN_NAMES_FOR_IC = 5
MIN_VALIDATION_PICKS = 30
PROBABILITY_GRID = (0.5, 0.55, 0.6, 0.65, 0.7)


@dataclass
class HorizonReport:
    horizon: int
    selected_model: str
    validation: dict
    holdout: dict
    notes: list[str] = field(default_factory=list)


def evaluate_horizon(
    frame: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    horizon: int,
    holdout_days: int = 126,
    test_days: int = 126,
    round_trip_cost_bps: float = 12.0,
) -> HorizonReport:
    """Fit on purged folds, freeze the choice, then score the holdout once."""
    labeled = frame.dropna(subset=["fwd_net", "mom_252_21"]).copy()
    labeled["date"] = pd.to_datetime(labeled["date"])
    dates = pd.DatetimeIndex(trading_index).sort_values().unique()
    cutoff = holdout_start(dates, holdout_days)
    folds = list(
        purged_walk_forward(dates, horizon, test_days=test_days, holdout_days=holdout_days)
    )
    if not folds:
        raise ValueError(f"No purged validation folds for horizon {horizon}")

    fold_rows = []
    logistic_scores = []
    for train_dates, test_dates in folds:
        train = labeled[labeled["date"].isin(train_dates)]
        test = labeled[labeled["date"].isin(test_dates)]
        if len(train) < 50 or test.empty:
            continue
        scores = _predict_all(train, test)
        fold_rows.append(_fold_metrics(test, scores))
        logistic_scores.append(scores[["date", "ticker", "fwd_net", "logistic_prob"]])

    if not fold_rows:
        raise ValueError(f"Validation folds were empty for horizon {horizon}")
    validation = _average_folds(fold_rows)
    selected = _select_regressor(validation)
    threshold, threshold_note = _select_threshold(logistic_scores)

    train_dates = pre_holdout_train_dates(dates, horizon, holdout_days)
    train = labeled[labeled["date"].isin(train_dates)]
    holdout_labeled = labeled[labeled["date"] >= cutoff]
    holdout_scores = _predict_all(train, holdout_labeled) if not holdout_labeled.empty else pd.DataFrame()
    holdout = _score_block(holdout_scores, selected, threshold, round_trip_cost_bps) if not holdout_scores.empty else {}
    holdout["start"] = str(pd.Timestamp(cutoff).date())
    holdout["end"] = str(pd.Timestamp(dates[-1]).date()) if len(dates) else None
    notes = [
        f"Selected {selected} by mean validation Spearman IC before {pd.Timestamp(cutoff).date()}.",
        threshold_note,
        "Holdout rows were not used to choose the model or the probability threshold.",
        "Precision is the share of recommendations with a positive net holding-period return, not the share of up days in a portfolio.",
    ]
    return HorizonReport(
        horizon=horizon,
        selected_model=selected,
        validation=validation,
        holdout=holdout,
        notes=notes,
    )


def _predict_all(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    medians = train[FEATURE_COLUMNS].median()
    x_train = train[FEATURE_COLUMNS].fillna(medians)
    x_test = test[FEATURE_COLUMNS].fillna(medians)
    out = test[["date", "ticker", "fwd_net", "mom_252_21", "dist_sma_200"]].copy()
    out["momentum"] = test["mom_252_21"].to_numpy()
    out["ridge"] = _ridge_predict(x_train, train["fwd_net"], x_test)
    out["boosted_tree"] = _tree_predict(x_train, train["fwd_net"], x_test)
    out["logistic_prob"] = _logistic_predict(x_train, train["fwd_net"] > 0, x_test)
    return out


def _ridge_predict(x_train, y_train, x_test) -> np.ndarray:
    mean = x_train.mean()
    scale = x_train.std(ddof=0).replace(0, 1.0)
    model = Ridge(alpha=1.0)
    model.fit((x_train - mean) / scale, y_train)
    return model.predict((x_test - mean) / scale)


def _tree_predict(x_train, y_train, x_test) -> np.ndarray:
    model = HistGradientBoostingRegressor(
        max_depth=3,
        max_iter=80,
        learning_rate=0.05,
        random_state=0,
    )
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _logistic_predict(x_train, y_positive, x_test) -> np.ndarray:
    if y_positive.nunique() < 2:
        return np.full(len(x_test), np.nan)
    mean = x_train.mean()
    scale = x_train.std(ddof=0).replace(0, 1.0)
    model = LogisticRegression(C=1.0, max_iter=200, random_state=0)
    model.fit((x_train - mean) / scale, y_positive.astype(int))
    return model.predict_proba((x_test - mean) / scale)[:, 1]


def _fold_metrics(test: pd.DataFrame, scores: pd.DataFrame) -> dict:
    row = {"n_rows": int(len(test))}
    for name in ("momentum", "ridge", "boosted_tree"):
        row[f"ic_{name}"] = _mean_daily_spearman(scores, name)
    return row


def mean_daily_feature_ics(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """Mean daily Spearman IC of each feature against net forward return.

    ``dates`` chooses the sample. Pass pre-holdout dates to describe development
    data, and holdout dates only when reporting the untouched window separately.
    """
    allowed = set(pd.DatetimeIndex(dates))
    subset = frame[frame["date"].isin(allowed)]
    return {column: _mean_daily_spearman(subset, column) for column in FEATURE_COLUMNS}


def _mean_daily_spearman(scores: pd.DataFrame, column: str) -> float:
    values = []
    for _, day in scores.groupby("date"):
        if len(day) < MIN_NAMES_FOR_IC:
            continue
        pair = day[[column, "fwd_net"]].dropna()
        if len(pair) < MIN_NAMES_FOR_IC or pair[column].nunique() < 2 or pair["fwd_net"].nunique() < 2:
            continue
        values.append(float(pair[column].corr(pair["fwd_net"], method="spearman")))
    if not values:
        return float("nan")
    return float(np.mean(values))


def _average_folds(rows: list[dict]) -> dict:
    frame = pd.DataFrame(rows)
    out = {"n_folds": int(len(frame)), "n_rows": int(frame["n_rows"].sum())}
    for column in frame.columns:
        if column.startswith("ic_"):
            out[column] = float(frame[column].mean())
    return out


def _select_regressor(validation: dict) -> str:
    """Highest mean validation IC. Ties prefer the simpler score."""
    order = ("momentum", "ridge", "boosted_tree")
    best_name = "momentum"
    best_value = float("-inf")
    for name in order:
        value = validation.get(f"ic_{name}", float("nan"))
        if value is None or not np.isfinite(value):
            continue
        if value > best_value:
            best_value = float(value)
            best_name = name
    return best_name


def _select_threshold(blocks: list[pd.DataFrame]) -> tuple[float | None, str]:
    if not blocks:
        return None, "No logistic scores were produced; the probability rule recommends nobody."
    pooled = pd.concat(blocks, ignore_index=True).dropna(subset=["logistic_prob", "fwd_net"])
    best = None
    best_precision = -1.0
    for threshold in PROBABILITY_GRID:
        picked = pooled[pooled["logistic_prob"] >= threshold]
        if len(picked) < MIN_VALIDATION_PICKS:
            continue
        precision = float((picked["fwd_net"] > 0).mean())
        if precision > best_precision:
            best_precision = precision
            best = float(threshold)
    if best is None:
        return None, (
            f"No probability threshold in {PROBABILITY_GRID} produced {MIN_VALIDATION_PICKS} "
            "validation recommendations. The probability rule recommends nobody."
        )
    return best, (
        f"Logistic threshold {best:.2f} was frozen from validation "
        f"(precision {best_precision:.3f} on the picks used to choose it)."
    )


def _score_block(
    scores: pd.DataFrame,
    selected: str,
    threshold: float | None,
    round_trip_cost_bps: float,
) -> dict:
    min_score = float(round_trip_cost_bps) / 10_000.0
    recommendations = _recommendations(scores, selected, min_score)
    precision, n_picks, mean_net = _precision(recommendations)
    base = scores["fwd_net"].dropna()
    out = {
        "model": selected,
        "ic": _mean_daily_spearman(scores, selected),
        "pooled_spearman": _pooled_spearman(scores, selected),
        "n_labeled_rows": int(scores["fwd_net"].notna().sum()),
        "n_dates": int(scores["date"].nunique()),
        "recommendation_count": n_picks,
        "precision_profitable": precision,
        "mean_recommended_net_return": mean_net,
        "base_rate_positive_net": float((base > 0).mean()) if len(base) else float("nan"),
        "mean_net_return_all_names": float(base.mean()) if len(base) else float("nan"),
        "logistic_threshold": threshold,
    }
    if threshold is None:
        out["logistic_precision_profitable"] = float("nan")
        out["logistic_recommendation_count"] = 0
    else:
        logistic_picks = _logistic_recommendations(scores, threshold)
        prec, count, mean_log = _precision(logistic_picks)
        out["logistic_precision_profitable"] = prec
        out["logistic_recommendation_count"] = count
        out["logistic_mean_net_return"] = mean_log
    return out


def _recommendations(scores: pd.DataFrame, model: str, min_score: float) -> pd.DataFrame:
    if model == "momentum":
        eligible = scores[(scores["mom_252_21"] > 0) & (scores["dist_sma_200"] > 0)].copy()
        score_col = "momentum"
    else:
        eligible = scores[scores[model] > min_score].copy()
        score_col = model
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(["date", score_col], ascending=[True, False])
    return eligible.groupby("date", as_index=False).head(1)


def _logistic_recommendations(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    eligible = scores[scores["logistic_prob"] >= threshold].copy()
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(["date", "logistic_prob"], ascending=[True, False])
    return eligible.groupby("date", as_index=False).head(1)


def _precision(picks: pd.DataFrame) -> tuple[float, int, float]:
    if picks is None or picks.empty:
        return float("nan"), 0, float("nan")
    realized = picks["fwd_net"].dropna()
    if realized.empty:
        return float("nan"), 0, float("nan")
    return float((realized > 0).mean()), int(len(realized)), float(realized.mean())


def _pooled_spearman(scores: pd.DataFrame, column: str) -> float:
    pair = scores[[column, "fwd_net"]].dropna()
    if len(pair) < MIN_NAMES_FOR_IC or pair[column].nunique() < 2:
        return float("nan")
    return float(pair[column].corr(pair["fwd_net"], method="spearman"))


def rank_latest(
    scores: pd.DataFrame,
    model: str,
    min_score: float,
    threshold: float | None,
) -> pd.DataFrame:
    """Rank the latest date. Returns an empty frame when nothing clears the bar."""
    if scores.empty:
        return scores
    latest = scores["date"].max()
    day = scores[scores["date"] == latest].copy()
    if model == "logistic":
        if threshold is None:
            return day.iloc[0:0]
        picked = day[day["logistic_prob"] >= threshold].sort_values("logistic_prob", ascending=False)
        return picked.head(1)
    return _recommendations(day, model, min_score)
