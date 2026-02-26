"""
Yahoo Finance real market data loader (via yfinance).

Fetches daily OHLCV data using the yfinance library — no API key required.
Supports any date range including current data through today.

Tickers are downloaded in a single batch call (yfinance handles concurrency
internally) and assembled into (prices, volume, benchmark) DataFrames matching
the format expected by the rest of the momentum model pipeline.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_yfinance_data(
    tickers: List[str],
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
    min_history_days: int = 252,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Fetch real daily OHLCV data from Yahoo Finance for a list of US tickers.

    Parameters
    ----------
    tickers : list[str]
        US stock ticker symbols (e.g. ['AAPL', 'MSFT', 'NVDA']).
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str | None
        End date in 'YYYY-MM-DD' format.  Defaults to today.
    min_history_days : int
        Tickers with fewer trading days than this are dropped.

    Returns
    -------
    prices    : pd.DataFrame  (Date × Ticker, adjusted-close)
    volume    : pd.DataFrame  (Date × Ticker, share volume)
    benchmark : pd.Series     (equal-weight index of all returned stocks)
    """
    import yfinance as yf

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    logger.info(
        "Fetching %d tickers from Yahoo Finance (%s → %s) …",
        len(tickers), start_date, end_date,
    )

    # Download all tickers in one batch; auto_adjust=True gives adj-close in "Close"
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError(
            "yfinance returned no data. This is usually caused by a network proxy "
            "blocking Yahoo Finance. Try running from a machine with direct internet access."
        )

    # yfinance returns a MultiIndex (field, ticker) when len(tickers) > 1
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
        volume = raw["Volume"].copy()
    else:
        # Single ticker — flatten
        ticker = tickers[0]
        prices = raw[["Close"]].rename(columns={"Close": ticker})
        volume = raw[["Volume"]].rename(columns={"Volume": ticker})

    prices.index = pd.DatetimeIndex(prices.index)
    volume.index = pd.DatetimeIndex(volume.index)

    # Report failed tickers
    failed = [t for t in tickers if t not in prices.columns or prices[t].isna().all()]
    if failed:
        logger.warning(
            "yfinance: failed to download %d/%d tickers: %s",
            len(failed), len(tickers), sorted(failed)[:20],
        )

    # Forward-fill price gaps, zero-fill missing volume
    prices = prices.ffill().bfill()
    volume = volume.fillna(0)

    # Drop tickers with insufficient history
    valid = prices.count() >= min_history_days
    dropped = valid[~valid].index.tolist()
    if dropped:
        logger.warning(
            "Dropping %d tickers with fewer than %d trading days.",
            len(dropped), min_history_days,
        )
    prices = prices.loc[:, valid]
    volume = volume.loc[:, valid]

    if prices.empty:
        raise RuntimeError(
            f"No tickers survived the minimum history filter ({min_history_days} days). "
            "Try a longer date range or lower --start date."
        )

    # Equal-weight benchmark
    ret = prices.pct_change().dropna(how="all")
    benchmark = (1 + ret.mean(axis=1)).cumprod() * 100
    benchmark = benchmark.rename("SP500_EW_YFinance")

    logger.info(
        "yfinance: loaded %d tickers × %d trading days  (%s → %s)",
        prices.shape[1],
        prices.shape[0],
        prices.index[0].date(),
        prices.index[-1].date(),
    )
    return prices, volume, benchmark
