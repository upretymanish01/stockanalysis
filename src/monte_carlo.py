"""Monte Carlo simulations for stock prices and portfolios.

Implements Geometric Brownian Motion (GBM) for price path simulation
and derives risk metrics (VaR, CVaR, probability of loss, percentiles).
"""

from typing import Optional

import numpy as np
import pandas as pd


def geometric_brownian_motion(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    steps: int,
    n_simulations: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate stock price paths using Geometric Brownian Motion.

    dS = mu * S * dt + sigma * S * dW

    Args:
        S0: Initial price
        mu: Annualized drift (expected return)
        sigma: Annualized volatility
        T: Time horizon in years
        steps: Number of time steps
        n_simulations: Number of paths to simulate
        seed: Random seed for reproducibility

    Returns:
        Array of shape (n_simulations, steps + 1) containing price paths
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / steps
    # Pre-compute constants
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Random shocks
    Z = np.random.normal(size=(n_simulations, steps))

    # Log returns
    log_returns = drift + diffusion * Z

    # Cumulative log returns and prices
    log_prices = np.cumsum(log_returns, axis=1)
    prices = S0 * np.exp(log_prices)

    # Prepend initial price
    paths = np.column_stack([np.full(n_simulations, S0), prices])
    return paths


def simulate_stock_paths(
    S0: float,
    mu: float,
    sigma: float,
    days: int = 252,
    n_simulations: int = 10_000,
    trading_days_per_year: int = 252,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper that returns paths as a DataFrame.

    Columns are simulation index, rows are time steps (day 0 to day `days`).
    """
    T = days / trading_days_per_year
    paths = geometric_brownian_motion(
        S0=S0,
        mu=mu,
        sigma=sigma,
        T=T,
        steps=days,
        n_simulations=n_simulations,
        seed=seed,
    )
    # Transpose so each column is a path (easier for plotting)
    df = pd.DataFrame(
        paths.T,
        columns=[f"path_{i}" for i in range(n_simulations)],
    )
    df.index.name = "day"
    return df


def summarize_terminal_prices(paths: np.ndarray | pd.DataFrame) -> dict:
    """
    Summarize the distribution of terminal (final) prices.

    Args:
        paths: Array of shape (n_sim, steps+1) or DataFrame from simulate_stock_paths

    Returns:
        Dictionary with mean, std, percentiles, probability of loss, VaR, CVaR
    """
    if isinstance(paths, pd.DataFrame):
        terminal = paths.iloc[-1].values
        S0 = paths.iloc[0].iloc[0]
    else:
        terminal = paths[:, -1]
        S0 = paths[0, 0]

    returns = (terminal - S0) / S0

    # Percentiles of terminal price
    percentiles = np.percentile(terminal, [5, 10, 25, 50, 75, 90, 95])

    # Risk metrics (on returns)
    var_95 = np.percentile(returns, 5)          # 5% left-tail
    cvar_95 = returns[returns <= var_95].mean()  # Expected shortfall

    return {
        "S0": float(S0),
        "mean_terminal": float(np.mean(terminal)),
        "std_terminal": float(np.std(terminal)),
        "median_terminal": float(np.median(terminal)),
        "p5": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p90": float(percentiles[5]),
        "p95": float(percentiles[6]),
        "prob_loss": float(np.mean(terminal < S0)),
        "prob_gain": float(np.mean(terminal > S0)),
        "var_95": float(var_95),          # 95% VaR (as return)
        "cvar_95": float(cvar_95),        # 95% CVaR (as return)
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
    }


def estimate_params_from_history(
    prices: pd.Series,
    trading_days_per_year: int = 252,
) -> dict:
    """
    Estimate annualized mu and sigma from historical price series.

    Uses log returns.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu_daily = log_returns.mean()
    sigma_daily = log_returns.std()

    mu = mu_daily * trading_days_per_year
    sigma = sigma_daily * np.sqrt(trading_days_per_year)

    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "mu_daily": float(mu_daily),
        "sigma_daily": float(sigma_daily),
        "n_observations": int(len(log_returns)),
    }
