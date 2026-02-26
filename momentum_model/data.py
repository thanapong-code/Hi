"""
Data fetching and preprocessing for the momentum model.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Downloads and preprocesses OHLCV price data for a list of US tickers.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to fetch.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str | None
        End date in 'YYYY-MM-DD' format, or None for today.
    min_history_days : int
        Minimum number of trading days required; shorter series are dropped.
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        min_history_days: int = 252,
    ) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.min_history_days = min_history_days

        self._prices: Optional[pd.DataFrame] = None
        self._volume: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(self) -> "DataFetcher":
        """Download data from Yahoo Finance and run quality checks."""
        logger.info("Fetching data for %d tickers…", len(self.tickers))

        raw = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
            volume = raw["Volume"].copy()
        else:
            # Single ticker returned as flat DataFrame
            prices = raw[["Close"]].copy()
            prices.columns = self.tickers[:1]
            volume = raw[["Volume"]].copy()
            volume.columns = self.tickers[:1]

        prices = self._clean(prices)
        volume = self._clean(volume, fill_method="zero")

        # Drop tickers with insufficient history
        valid = prices.count() >= self.min_history_days
        dropped = valid[~valid].index.tolist()
        if dropped:
            logger.warning("Dropping %d tickers with insufficient history: %s", len(dropped), dropped)
        prices = prices.loc[:, valid]
        volume = volume.loc[:, valid]

        self._prices = prices
        self._volume = volume
        self._returns = prices.pct_change()

        logger.info(
            "Loaded %d tickers × %d trading days (%.1f%% coverage)",
            prices.shape[1],
            prices.shape[0],
            100 * prices.notna().mean().mean(),
        )
        return self

    @property
    def prices(self) -> pd.DataFrame:
        self._check_fetched()
        return self._prices

    @property
    def volume(self) -> pd.DataFrame:
        self._check_fetched()
        return self._volume

    @property
    def returns(self) -> pd.DataFrame:
        self._check_fetched()
        return self._returns

    @property
    def tickers_available(self) -> List[str]:
        self._check_fetched()
        return self._prices.columns.tolist()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame, fill_method: str = "ffill") -> pd.DataFrame:
        """Forward-fill gaps then back-fill leading NaNs."""
        df = df.sort_index()
        if fill_method == "zero":
            return df.fillna(0)
        return df.ffill().bfill()

    def _check_fetched(self) -> None:
        if self._prices is None:
            raise RuntimeError("Call .fetch() first.")

    # ── Derived datasets ──────────────────────────────────────────────────────

    def rolling_returns(self, window: int) -> pd.DataFrame:
        """Cumulative return over the last `window` trading days."""
        return self.prices.pct_change(window)

    def rolling_volatility(self, window: int = 21, annualise: bool = True) -> pd.DataFrame:
        """Rolling annualised volatility."""
        vol = self.returns.rolling(window).std()
        if annualise:
            vol = vol * np.sqrt(252)
        return vol

    def rolling_volume_ratio(self, window: int = 20) -> pd.DataFrame:
        """Current volume vs. rolling average volume (> 1 = above average)."""
        avg_vol = self.volume.rolling(window).mean()
        return self.volume / avg_vol.replace(0, np.nan)

    def get_market_cap_proxy(self) -> pd.Series:
        """
        Simple market-cap proxy: last price × last non-zero volume.
        Useful for weighting without a separate fundamentals feed.
        """
        last_price = self.prices.iloc[-1]
        avg_volume = self.volume.mean()
        return last_price * avg_volume

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame with key statistics for each ticker."""
        stats = pd.DataFrame(index=self.tickers_available)
        stats["first_date"] = self.prices.apply(lambda s: s.first_valid_index())
        stats["last_date"] = self.prices.apply(lambda s: s.last_valid_index())
        stats["trading_days"] = self.prices.count()
        stats["last_price"] = self.prices.iloc[-1]
        annual_ret = (1 + self.returns).prod() ** (252 / self.returns.count()) - 1
        stats["annual_return"] = annual_ret
        stats["annual_vol"] = self.returns.std() * np.sqrt(252)
        stats["sharpe"] = stats["annual_return"] / stats["annual_vol"]
        return stats.round(4)
