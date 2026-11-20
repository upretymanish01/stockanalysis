"""Tests for Monte Carlo module."""

import numpy as np
import pandas as pd

from src.monte_carlo import (
    geometric_brownian_motion,
    simulate_stock_paths,
    summarize_terminal_prices,
    estimate_params_from_history,
)


def test_gbm_shape_and_initial_price():
    paths = geometric_brownian_motion(
        S0=100.0,
        mu=0.1,
        sigma=0.2,
        T=1.0,
        steps=252,
        n_simulations=1000,
        seed=42,
    )
    assert paths.shape == (1000, 253)
    assert np.allclose(paths[:, 0], 100.0)


def test_simulate_stock_paths_dataframe():
    df = simulate_stock_paths(
        S0=50.0,
        mu=0.08,
        sigma=0.25,
        days=126,
        n_simulations=500,
        seed=123,
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (127, 500)
    assert df.iloc[0].iloc[0] == 50.0


def test_summarize_terminal_prices():
    paths = geometric_brownian_motion(
        S0=100.0, mu=0.1, sigma=0.2, T=1.0, steps=252, n_simulations=5000, seed=7
    )
    summary = summarize_terminal_prices(paths)
    assert summary["S0"] == 100.0
    assert 0.0 <= summary["prob_loss"] <= 1.0
    assert summary["p5"] < summary["p50"] < summary["p95"]
    assert summary["var_95"] < 0  # left tail should be negative in most cases


def test_estimate_params_from_history():
    # Synthetic lognormal prices
    rng = np.random.default_rng(42)
    log_returns = rng.normal(0.0003, 0.015, 500)
    prices = pd.Series(100 * np.exp(np.cumsum(log_returns)))

    params = estimate_params_from_history(prices)
    assert params["n_observations"] == 499
    assert params["sigma"] > 0
    assert "mu" in params
