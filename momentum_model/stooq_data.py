"""
Stooq.com real market data loader.

Fetches daily OHLCV data directly from Stooq's free CSV download endpoint —
no API key required.  Works for any date range Stooq has coverage for,
including current data through today.

URL pattern (per ticker):
    https://stooq.com/q/d/l/?s={TICKER}.US&d1={YYYYMMDD}&d2={YYYYMMDD}&i=d

Tickers are fetched concurrently (ThreadPoolExecutor) and assembled into
(prices, volume, benchmark) DataFrames matching the format expected by the
rest of the momentum model pipeline.
"""

from __future__ import annotations

import io
import logging
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://stooq.com/q/d/l/?s={ticker}.US&d1={d1}&d2={d2}&i=d"
_MAX_WORKERS = 10  # concurrent downloads
_TIMEOUT = 20      # seconds per request


def _fmt_date(dt: str | date) -> str:
    """Convert a date or 'YYYY-MM-DD' string to Stooq's 'YYYYMMDD' format."""
    if isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d").date()
    return dt.strftime("%Y%m%d")


def _fetch_one(ticker: str, d1: str, d2: str) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Download a single ticker's daily OHLCV from Stooq.

    Returns
    -------
    (ticker, DataFrame | None)
        DataFrame has a DatetimeIndex and columns [Open, High, Low, Close, Volume].
        Returns None on any network or parse error.
    """
    url = _BASE_URL.format(ticker=ticker.upper(), d1=d1, d2=d2)
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "python/3.11"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT, context=ctx) as resp:
            raw = resp.read()

        df = pd.read_csv(io.BytesIO(raw), parse_dates=["Date"], index_col="Date")

        # Stooq sometimes returns an HTML error page instead of CSV
        if df.empty or "Close" not in df.columns:
            return ticker, None

        df.index = pd.DatetimeIndex(df.index)
        df = df.sort_index()
        return ticker, df

    except Exception as exc:
        logger.debug("Stooq fetch failed for %s: %s", ticker, exc)
        return ticker, None


def fetch_stooq_data(
    tickers: List[str],
    start_date: str = "2024-01-01",
    end_date: Optional[str] = None,
    min_history_days: int = 252,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Fetch real daily OHLCV data from Stooq.com for a list of US tickers.

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
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    d1 = _fmt_date(start_date)
    d2 = _fmt_date(end_date)

    logger.info(
        "Fetching %d tickers from Stooq (%s → %s) …", len(tickers), start_date, end_date
    )

    close_dict: dict[str, pd.Series] = {}
    volume_dict: dict[str, pd.Series] = {}
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, t, d1, d2): t for t in tickers}
        for fut in as_completed(futures):
            ticker, df = fut.result()
            if df is not None and not df.empty:
                close_dict[ticker] = df["Close"]
                volume_dict[ticker] = df["Volume"].fillna(0)
            else:
                failed.append(ticker)

    if failed:
        logger.warning(
            "Stooq: failed to download %d/%d tickers: %s",
            len(failed), len(tickers), sorted(failed)[:20],
        )

    if not close_dict:
        raise RuntimeError(
            "Stooq returned no data. This is usually caused by a network proxy "
            "blocking external financial data sites. Try running from a machine "
            "with direct internet access."
        )

    prices = pd.DataFrame(close_dict).sort_index()
    volume = pd.DataFrame(volume_dict).sort_index().fillna(0)

    # Align indices
    prices, volume = prices.align(volume, join="inner", axis=0)

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
    benchmark = benchmark.rename("SP500_EW_Stooq")

    logger.info(
        "Stooq: loaded %d tickers × %d trading days  (%s → %s)",
        prices.shape[1],
        prices.shape[0],
        prices.index[0].date(),
        prices.index[-1].date(),
    )
    return prices, volume, benchmark
