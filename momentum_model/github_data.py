"""
GitHub-hosted real market data loader.

Downloads the S&P 500 5-year daily OHLCV dataset (2013-02-08 → 2018-02-07)
from the plotly/datasets public repository on GitHub — accessible without
financial-data API access.

Source: https://github.com/plotly/datasets/blob/master/all_stocks_5yr.csv
  505 S&P 500 constituents · ~1,226 trading days each · columns: date,
  open, high, low, close, volume, Name
"""

from __future__ import annotations

import io
import logging
import ssl
import urllib.request
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_URL = "https://raw.githubusercontent.com/plotly/datasets/master/all_stocks_5yr.csv"


def fetch_github_sp500(
    tickers: Optional[list] = None,
    min_history_days: int = 252,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Download the S&P 500 5-year dataset from GitHub and return
    (prices, volume, benchmark) ready for the momentum model.

    Parameters
    ----------
    tickers : list | None
        Subset of tickers to keep.  None = all 505.
    min_history_days : int
        Drop tickers with fewer trading days than this.

    Returns
    -------
    prices    : pd.DataFrame  (Date × Ticker, adjusted close proxy)
    volume    : pd.DataFrame  (Date × Ticker, share volume)
    benchmark : pd.Series     (equal-weight index of all returned stocks)
    """
    logger.info("Downloading S&P 500 dataset from GitHub (plotly/datasets)…")

    ctx = ssl.create_default_context()
    req = urllib.request.Request(_URL, headers={"User-Agent": "python/3.11"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        raw = r.read()

    df = pd.read_csv(io.BytesIO(raw), parse_dates=["date"])
    df = df.rename(columns={"date": "Date", "close": "Close",
                             "volume": "Volume", "Name": "Ticker"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    if tickers:
        available = set(df["Ticker"].unique())
        tickers = [t for t in tickers if t in available]
        if not tickers:
            raise ValueError("None of the requested tickers are in the dataset.")
        df = df[df["Ticker"].isin(tickers)]

    # Pivot to wide format
    prices = df.pivot(index="Date", columns="Ticker", values="Close")
    volume = df.pivot(index="Date", columns="Ticker", values="Volume").fillna(0)

    prices.index = pd.DatetimeIndex(prices.index)
    volume.index = pd.DatetimeIndex(volume.index)

    # Forward-fill gaps, drop tickers with insufficient history
    prices = prices.ffill().bfill()
    volume = volume.ffill().fillna(0)

    valid = prices.count() >= min_history_days
    dropped = valid[~valid].index.tolist()
    if dropped:
        logger.warning("Dropping %d tickers with insufficient history.", len(dropped))
    prices = prices.loc[:, valid]
    volume = volume.loc[:, valid]

    # Equal-weight benchmark
    ret = prices.pct_change().dropna(how="all")
    bench = (1 + ret.mean(axis=1)).cumprod() * 100
    bench = bench.rename("SP500_EW")

    logger.info(
        "Loaded %d tickers × %d trading days  (%s → %s)",
        prices.shape[1], prices.shape[0],
        prices.index[0].date(), prices.index[-1].date(),
    )
    return prices, volume, bench
