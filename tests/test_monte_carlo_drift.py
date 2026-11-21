"""Regression tests for the double Ito adjustment on Monte Carlo drift."""

import numpy as np
import pandas as pd
import pytest

from src.monte_carlo import (
    estimate_params_from_history,
    geometric_brownian_motion,
)


def _exact_gbm_prices(
    mu: float,
    sigma: float,
    n: int = 2520,
    s0: float = 100.0,
    trading_days: int = 252,
) -> pd.Series:
    """Prices whose log returns have this exact sample mean and sample std.

    The series starts at s0, so estimating from the series recovers every
    constructed log return (nothing is absorbed into an unseen prior close).
    """
    if n % 2:
        raise ValueError("n must be even so the shocks are symmetric")
    dt = 1.0 / trading_days
    log_mean = (mu - 0.5 * sigma**2) * dt
    log_std = sigma * np.sqrt(dt)
    shocks = np.tile(np.array([1.0, -1.0]), n // 2)
    shocks = (shocks - shocks.mean()) / np.std(shocks, ddof=1)
    log_returns = log_mean + log_std * shocks
    levels = s0 * np.exp(np.cumsum(log_returns))
    return pd.Series(np.concatenate([[s0], levels]))


def test_estimate_params_returns_arithmetic_drift_not_log_drift():
    mu = 0.12
    sigma = 0.50  # 0.5 * sigma^2 = 0.125, the reported ~12%/yr bias
    prices = _exact_gbm_prices(mu, sigma)

    params = estimate_params_from_history(prices)

    assert params["sigma"] == pytest.approx(sigma, abs=1e-9)
    assert params["mu"] == pytest.approx(mu, abs=1e-9)
    # Mean log return stays lower by the single Ito term. Returning it as mu
    # is the bug: the simulator subtracts 0.5*sigma^2 again.
    assert params["mu_log"] == pytest.approx(mu - 0.5 * sigma**2, abs=1e-9)
    assert params["mu"] - params["mu_log"] == pytest.approx(0.5 * params["sigma"] ** 2)


def test_gbm_terminal_mean_matches_arithmetic_drift():
    """The simulator applies -0.5*sigma^2 once. E[S_T] = S0 * exp(mu * T)."""
    mu = 0.12
    sigma = 0.50
    s0 = 100.0
    horizon = 1.0
    paths = geometric_brownian_motion(
        S0=s0,
        mu=mu,
        sigma=sigma,
        T=horizon,
        steps=252,
        n_simulations=20_000,
        seed=1,
    )
    mean_growth = paths[:, -1].mean() / s0
    mean_log_growth = np.log(paths[:, -1] / s0).mean()

    assert mean_growth == pytest.approx(np.exp(mu * horizon), abs=0.02)
    assert mean_log_growth == pytest.approx((mu - 0.5 * sigma**2) * horizon, abs=0.015)


def test_chained_estimate_and_simulate_matches_input_drift():
    """Synthetic GBM -> estimated drift -> resimulated mean stays on that drift.

    Before the fix the chained mean return was about exp(mu - 0.5*sigma^2) - 1
    (~ -0.5% here) instead of exp(mu) - 1 (~ +12.7%).
    """
    mu = 0.12
    sigma = 0.50
    horizon = 1.0
    prices = _exact_gbm_prices(mu, sigma)
    params = estimate_params_from_history(prices)

    paths = geometric_brownian_motion(
        S0=100.0,
        mu=params["mu"],
        sigma=params["sigma"],
        T=horizon,
        steps=252,
        n_simulations=20_000,
        seed=1,
    )
    mean_return = paths[:, -1].mean() / 100.0 - 1.0
    expected = np.exp(mu * horizon) - 1.0
    double_counted = np.exp((mu - 0.5 * sigma**2) * horizon) - 1.0

    assert mean_return == pytest.approx(expected, abs=0.03)
    assert abs(mean_return - expected) < abs(mean_return - double_counted)
