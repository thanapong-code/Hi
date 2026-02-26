"""
Multi-factor momentum signal generation.

Signals produced
----------------
1. price_momentum  – cross-sectional rank of risk-adjusted price return
2. rsi_signal      – RSI-based overbought/oversold filtered momentum score
3. macd_signal     – MACD histogram normalised to [-1, 1]
4. volume_signal   – volume trend (price × volume flow)
5. volatility_adj  – volatility-adjusted (inverse-vol) momentum
6. rs_rating       – IBD-style Relative Strength Rating (12-month weighted return rank)

Hard filter (not a signal weight):
  EMA-200 filter  – stocks trading below their 200-day EMA are excluded from
                    the long portfolio entirely (composite set to NaN).

Each signal is cross-sectionally z-scored and then combined into a composite.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Low-level indicator helpers ───────────────────────────────────────────────

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
          ) -> pd.DataFrame:
    fast_ema = _ema(series, fast)
    slow_ema = _ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": histogram},
                        index=series.index)


def _bollinger_position(series: pd.Series, period: int = 20, n_std: float = 2.0
                        ) -> pd.Series:
    """Position within Bollinger Bands: 0 = lower band, 1 = upper band."""
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return (series - lower) / (upper - lower + 1e-10)


def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score (demean and scale each row)."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def _winsorise(df: pd.DataFrame, lower: float = -3.0, upper: float = 3.0
               ) -> pd.DataFrame:
    return df.clip(lower=lower, upper=upper)


# ── Main class ────────────────────────────────────────────────────────────────

class MomentumSignals:
    """
    Computes momentum signals for a universe of stocks.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices (rows = dates, columns = tickers).
    volume : pd.DataFrame
        Share volume (same shape as prices).
    signal_weights : dict[str, float] | None
        Weights for combining sub-signals.  Must sum to 1.
    momentum_windows : dict[str, int] | None
        Lookback windows for short / medium / long momentum.
    skip_recent : int
        Skip the most-recent N days when computing price momentum
        (avoids short-term reversal contamination).
    """

    DEFAULT_WEIGHTS = {
        "price_momentum": 0.35,
        "rsi_signal":     0.10,
        "macd_signal":    0.10,
        "volume_signal":  0.10,
        "volatility_adj": 0.10,
        "rs_rating":      0.25,
    }

    DEFAULT_WINDOWS = {
        "short":  21,
        "medium": 63,
        "long":  252,
    }

    def __init__(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        signal_weights: Optional[Dict[str, float]] = None,
        momentum_windows: Optional[Dict[str, int]] = None,
        skip_recent: int = 21,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal_period: int = 9,
        ema_fast: int = 50,
        ema_slow: int = 200,
        ema_filter: bool = True,
    ) -> None:
        self.prices = prices
        self.volume = volume
        self.weights = signal_weights or self.DEFAULT_WEIGHTS
        self.windows = momentum_windows or self.DEFAULT_WINDOWS
        self.skip_recent = skip_recent
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_filter = ema_filter

        self._returns = prices.pct_change()
        self._composite: Optional[pd.DataFrame] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self) -> "MomentumSignals":
        """Compute all sub-signals and combine into a composite score."""
        logger.info("Computing momentum signals…")
        signals = {
            "price_momentum": self._price_momentum(),
            "rsi_signal":     self._rsi_signal(),
            "macd_signal":    self._macd_signal(),
            "volume_signal":  self._volume_signal(),
            "volatility_adj": self._volatility_adj_momentum(),
            "rs_rating":      self._rs_rating_signal(),
        }

        composite = pd.DataFrame(0.0, index=self.prices.index,
                                 columns=self.prices.columns)
        for name, sig in signals.items():
            w = self.weights.get(name, 0.0)
            sig_z = _winsorise(_cs_zscore(sig))
            composite += w * sig_z.reindex_like(composite).fillna(0)

        # ── EMA-200 hard filter ───────────────────────────────────────────────
        # Stocks trading below their EMA-200 are excluded from the long
        # universe by setting their composite score to NaN.  top_n() and
        # nlargest() naturally skip NaN, so no further changes are needed.
        if self.ema_filter:
            ema200 = self.prices.apply(lambda s: _ema(s, self.ema_slow))
            below_ema200 = self.prices < ema200
            n_masked = int(below_ema200.sum().sum())
            composite[below_ema200] = np.nan
            logger.info("EMA-%d hard filter: %d stock-days masked (below EMA).",
                        self.ema_slow, n_masked)

        self._composite = composite
        self._sub_signals = signals
        logger.info("Signals computed.")
        return self

    @property
    def composite(self) -> pd.DataFrame:
        """Combined momentum score (higher = stronger momentum)."""
        self._check_computed()
        return self._composite

    @property
    def sub_signals(self) -> Dict[str, pd.DataFrame]:
        """Individual signal DataFrames keyed by name."""
        self._check_computed()
        return self._sub_signals

    def ranks(self, date: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        Cross-sectional percentile rank for a given date.
        Returns a Series sorted descending (rank 1 = strongest momentum).
        """
        self._check_computed()
        row = self._composite.loc[date] if date else self._composite.iloc[-1]
        return row.rank(ascending=False, method="first").sort_values()

    def top_n(self, n: int = 20, date: Optional[pd.Timestamp] = None) -> pd.Index:
        """Return the top-N ticker symbols by composite momentum score."""
        self._check_computed()
        row = self._composite.loc[date] if date else self._composite.iloc[-1]
        return row.nlargest(n).index

    def bottom_n(self, n: int = 20, date: Optional[pd.Timestamp] = None) -> pd.Index:
        """Return the bottom-N ticker symbols (potential shorts)."""
        self._check_computed()
        row = self._composite.loc[date] if date else self._composite.iloc[-1]
        return row.nsmallest(n).index

    def signal_snapshot(self, date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        DataFrame of all sub-signals + composite for a given date,
        sorted by composite score descending.
        """
        self._check_computed()
        idx = date or self._composite.index[-1]
        snap = pd.DataFrame(
            {name: sig.loc[idx] for name, sig in self._sub_signals.items()}
        )
        snap["composite"] = self._composite.loc[idx]
        return snap.sort_values("composite", ascending=False).round(4)

    # ── Sub-signal implementations ────────────────────────────────────────────

    def _price_momentum(self) -> pd.DataFrame:
        """
        Blended price momentum:
          40% long (12-1 month, i.e. skip most-recent month)
          40% medium (3-month)
          20% short (1-month)
        """
        long_w  = self.windows["long"]
        med_w   = self.windows["medium"]
        short_w = self.windows["short"]
        skip    = self.skip_recent

        # Long: 12-month return, skip last month
        long_ret = (
            self.prices.shift(skip).pct_change(long_w - skip)
        )
        # Medium: 3-month return, skip last month
        med_ret = (
            self.prices.shift(skip).pct_change(med_w - skip)
        )
        # Short: 1-month return (no skip)
        short_ret = self.prices.pct_change(short_w)

        # Volatility-adjust each horizon
        vol = self._returns.rolling(long_w).std() * np.sqrt(252)
        long_adj  = long_ret  / vol.replace(0, np.nan)
        med_adj   = med_ret   / vol.replace(0, np.nan)
        short_adj = short_ret / vol.replace(0, np.nan)

        return 0.40 * long_adj + 0.40 * med_adj + 0.20 * short_adj

    def _rsi_signal(self) -> pd.DataFrame:
        """
        RSI-based signal: scores between 30–70 get positive weight.
        Extreme RSI (overbought > 80, oversold < 20) are penalised.
        """
        rsi_df = self.prices.apply(lambda s: _rsi(s, self.rsi_period))

        # Map RSI to a momentum-friendly score in [-1, 1]
        # RSI 50–70 → rising momentum; RSI 30–50 → fading
        score = (rsi_df - 50) / 50  # scale: -1 to +1
        # Penalise extremes: RSI > 80 or < 20 → reversion risk
        extreme_high = rsi_df > 80
        extreme_low  = rsi_df < 20
        score[extreme_high] *= -0.5
        score[extreme_low]  *= -0.5
        return score

    def _macd_signal(self) -> pd.DataFrame:
        """MACD histogram normalised by its rolling std."""
        result = {}
        for ticker in self.prices.columns:
            macd_data = _macd(
                self.prices[ticker],
                fast=self.macd_fast,
                slow=self.macd_slow,
                signal=self.macd_signal_period,
            )
            hist = macd_data["hist"]
            rolling_std = hist.rolling(63).std().replace(0, np.nan)
            result[ticker] = hist / rolling_std
        return pd.DataFrame(result)

    def _volume_signal(self) -> pd.DataFrame:
        """
        On-Balance Volume (OBV) momentum:
        Captures whether volume is flowing into (positive) or out of (negative)
        a stock, normalised cross-sectionally.
        """
        obv = {}
        for ticker in self.prices.columns:
            price_s = self.prices[ticker]
            vol_s   = self.volume[ticker]
            direction = np.sign(price_s.diff())
            obv_series = (direction * vol_s).cumsum()
            # OBV momentum: rate of change over medium window
            obv[ticker] = obv_series.pct_change(self.windows["medium"])
        return pd.DataFrame(obv)

    def _volatility_adj_momentum(self) -> pd.DataFrame:
        """
        Inverse-volatility adjusted momentum:
        Stocks with lower vol get a higher score for the same return,
        rewarding consistent trend-followers over volatile movers.
        """
        ret = self.prices.pct_change(self.windows["medium"])
        vol = self._returns.rolling(self.windows["medium"]).std() * np.sqrt(252)
        return ret / vol.replace(0, np.nan)

    def _rs_rating_signal(self) -> pd.DataFrame:
        """
        IBD-style Relative Strength (RS) Rating.

        Weights price performance over four lookback periods to emphasise
        recent momentum (same formula used by Investor's Business Daily):
          40% × 3-month return  (63 trading days)
          20% × 6-month return  (126 trading days)
          20% × 9-month return  (189 trading days)
          20% × 12-month return (252 trading days)

        The blended score is then ranked cross-sectionally so that the
        best-performing stock scores near 99 and the worst near 1.
        Cross-sectional z-scoring in compute() normalises further.
        """
        r3  = self.prices.pct_change(63)    # ~3 months
        r6  = self.prices.pct_change(126)   # ~6 months
        r9  = self.prices.pct_change(189)   # ~9 months
        r12 = self.prices.pct_change(252)   # ~12 months

        rs_score = 0.40 * r3 + 0.20 * r6 + 0.20 * r9 + 0.20 * r12

        # Percentile rank cross-sectionally (1 = weakest, 99 = strongest)
        rs_rank = rs_score.rank(axis=1, pct=True) * 98 + 1
        return rs_rank

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_computed(self) -> None:
        if self._composite is None:
            raise RuntimeError("Call .compute() first.")
