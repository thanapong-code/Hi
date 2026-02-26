"""
Synthetic market data generator for offline / testing use.

Generates correlated, realistic-looking stock price and volume series
using Geometric Brownian Motion with a common market factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional


def generate_market_data(
    tickers: List[str],
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    seed: int = 42,
    market_vol: float = 0.15,    # annualised market volatility
    idio_vol: float = 0.20,      # annualised idiosyncratic vol
    market_drift: float = 0.08,  # annualised market return
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate correlated daily OHLCV data.

    Returns
    -------
    prices : pd.DataFrame  (Date × Ticker, adjusted close)
    volume : pd.DataFrame  (Date × Ticker, share volume)
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(
        start=start_date,
        end=end_date or pd.Timestamp.today().strftime("%Y-%m-%d"),
    )
    T = len(dates)
    N = len(tickers)

    dt = 1 / 252
    sigma_m = market_vol * np.sqrt(dt)
    sigma_i = idio_vol  * np.sqrt(dt)
    mu_daily = market_drift * dt - 0.5 * (market_vol ** 2) * dt

    # Each ticker has its own beta (market sensitivity) ∈ [0.5, 1.8]
    betas = rng.uniform(0.5, 1.8, size=N)

    # Each ticker has its own drift bias ∈ [-2%, +6%] annualised
    alpha_daily = rng.uniform(-0.02, 0.06, size=N) * dt

    # Market factor shocks
    market_shocks = rng.normal(mu_daily, sigma_m, size=T)

    # Idiosyncratic shocks
    idio_shocks = rng.normal(0, sigma_i, size=(T, N))

    # Log-returns
    log_ret = market_shocks[:, None] * betas[None, :] + idio_shocks + alpha_daily[None, :]

    # Build price series (starting price ∈ $20–$500)
    start_prices = rng.uniform(20, 500, size=N)
    log_prices = np.cumsum(log_ret, axis=0)
    prices = start_prices[None, :] * np.exp(log_prices)

    price_df = pd.DataFrame(prices, index=dates, columns=tickers)

    # Volume: log-normal, mean-reverting around a base level
    # Base avg volume ∈ 1M–50M shares, correlated with price moves
    base_vol = rng.uniform(1_000_000, 50_000_000, size=N)
    vol_noise = rng.lognormal(0, 0.4, size=(T, N))
    # Volume spikes on large price moves
    price_move_mag = np.abs(log_ret) / sigma_i
    vol_multiplier = 1 + 0.5 * price_move_mag
    volume = base_vol[None, :] * vol_noise * vol_multiplier

    volume_df = pd.DataFrame(volume.astype(int), index=dates, columns=tickers)

    return price_df, volume_df


def generate_benchmark(
    price_df: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    seed: int = 0,
) -> pd.Series:
    """
    Generate an equal-weighted benchmark index from simulated prices,
    optionally adding small-cap bias noise.
    """
    if weights is None:
        weights = np.ones(price_df.shape[1]) / price_df.shape[1]
    returns = price_df.pct_change().dropna()
    port_ret = (returns * weights).sum(axis=1)
    bench = (1 + port_ret).cumprod() * 100  # index starts at 100
    return bench.rename("Benchmark")
