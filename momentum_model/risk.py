"""
Risk management utilities for the momentum model.

Provides:
  - Position-level stop-loss and trailing-stop monitoring
  - Portfolio-level risk checks (drawdown, concentration, leverage)
  - VaR / CVaR estimation (historical and parametric)
  - Correlation and factor-exposure monitoring
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ── Stop-loss tracker ─────────────────────────────────────────────────────────

@dataclass
class PositionState:
    ticker: str
    entry_price: float
    entry_date: pd.Timestamp
    peak_price: float = field(init=False)
    current_price: float = field(init=False)
    stop_triggered: bool = False
    stop_type: str = ""

    def __post_init__(self):
        self.peak_price = self.entry_price
        self.current_price = self.entry_price

    def update(self, price: float) -> None:
        self.current_price = price
        self.peak_price = max(self.peak_price, price)

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def drawdown_from_peak(self) -> float:
        if self.peak_price == 0:
            return 0.0
        return (self.current_price - self.peak_price) / self.peak_price


class StopLossManager:
    """
    Tracks open positions and fires stop-loss / trailing-stop alerts.

    Parameters
    ----------
    hard_stop_pct : float
        Exit if return from entry < -hard_stop_pct.
    trailing_stop_pct : float
        Exit if drawdown from peak < -trailing_stop_pct.
    """

    def __init__(self, hard_stop_pct: float = 0.10, trailing_stop_pct: float = 0.08) -> None:
        self.hard_stop = hard_stop_pct
        self.trail_stop = trailing_stop_pct
        self._positions: Dict[str, PositionState] = {}

    def open_position(self, ticker: str, price: float, date: pd.Timestamp) -> None:
        self._positions[ticker] = PositionState(ticker=ticker, entry_price=price, entry_date=date)

    def close_position(self, ticker: str) -> None:
        self._positions.pop(ticker, None)

    def update(self, price_row: pd.Series, date: pd.Timestamp) -> List[str]:
        """
        Update all positions with latest prices.

        Returns
        -------
        list[str]
            Tickers that should be exited due to stop triggers.
        """
        exits = []
        for ticker, pos in list(self._positions.items()):
            p = price_row.get(ticker, np.nan)
            if np.isnan(p):
                continue
            pos.update(p)

            if pos.pnl_pct < -self.hard_stop:
                pos.stop_triggered = True
                pos.stop_type = "hard_stop"
                logger.info("%s hard stop triggered (P&L: %.1f%%) on %s",
                            ticker, pos.pnl_pct * 100, date)
                exits.append(ticker)
            elif pos.drawdown_from_peak < -self.trail_stop:
                pos.stop_triggered = True
                pos.stop_type = "trailing_stop"
                logger.info("%s trailing stop triggered (drawdown from peak: %.1f%%) on %s",
                            ticker, pos.drawdown_from_peak * 100, date)
                exits.append(ticker)

        for ticker in exits:
            self.close_position(ticker)

        return exits

    def open_positions(self) -> pd.DataFrame:
        if not self._positions:
            return pd.DataFrame()
        rows = []
        for pos in self._positions.values():
            rows.append({
                "ticker": pos.ticker,
                "entry_date": pos.entry_date,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "peak_price": pos.peak_price,
                "pnl_pct": pos.pnl_pct,
                "drawdown_from_peak": pos.drawdown_from_peak,
            })
        return pd.DataFrame(rows).set_index("ticker").round(4)


# ── Portfolio risk checks ─────────────────────────────────────────────────────

class PortfolioRiskMonitor:
    """
    Real-time portfolio-level risk monitoring.

    Parameters
    ----------
    returns_history : pd.DataFrame
        Full daily returns panel (updated continuously).
    max_drawdown_limit : float
        Portfolio drawdown limit (circuit-breaker).
    max_concentration : float
        Maximum single-name weight allowed.
    max_sector_concentration : float
        Maximum combined weight per sector.
    var_confidence : float
        Confidence level for VaR estimation.
    var_window : int
        Look-back window for historical VaR (days).
    """

    def __init__(
        self,
        returns_history: pd.DataFrame,
        max_drawdown_limit: float = 0.20,
        max_concentration: float = 0.10,
        max_sector_concentration: float = 0.30,
        var_confidence: float = 0.95,
        var_window: int = 252,
    ) -> None:
        self.returns = returns_history
        self.max_dd = max_drawdown_limit
        self.max_conc = max_concentration
        self.max_sector_conc = max_sector_concentration
        self.var_conf = var_confidence
        self.var_window = var_window

    def portfolio_var(self, weights: pd.Series, as_of: pd.Timestamp) -> float:
        """Historical VaR of the portfolio at `var_confidence`."""
        hist = self.returns.loc[:as_of].iloc[-self.var_window:]
        port_ret = (hist[weights.index] * weights).sum(axis=1)
        return float(np.percentile(port_ret, (1 - self.var_conf) * 100))

    def portfolio_cvar(self, weights: pd.Series, as_of: pd.Timestamp) -> float:
        """Historical CVaR (Expected Shortfall) of the portfolio."""
        var = self.portfolio_var(weights, as_of)
        hist = self.returns.loc[:as_of].iloc[-self.var_window:]
        port_ret = (hist[weights.index] * weights).sum(axis=1)
        return float(port_ret[port_ret <= var].mean())

    def parametric_var(self, weights: pd.Series, as_of: pd.Timestamp,
                       holding_period: int = 1) -> float:
        """Parametric (Gaussian) VaR."""
        hist = self.returns.loc[:as_of].iloc[-self.var_window:]
        port_ret = (hist[weights.index] * weights).sum(axis=1)
        mu = port_ret.mean()
        sigma = port_ret.std()
        z = stats.norm.ppf(1 - self.var_conf)
        return float(mu + z * sigma * np.sqrt(holding_period))

    def concentration_check(self, weights: pd.Series) -> Dict[str, bool]:
        """
        Returns a dict of risk flags:
          - over_concentrated : any single name > max_conc
          - herfindahl_high   : HHI > 0.10 (effective <10 stocks)
        """
        over = (weights > self.max_conc).any()
        hhi = (weights ** 2).sum()
        return {
            "over_concentrated": bool(over),
            "herfindahl_high": bool(hhi > 0.10),
            "herfindahl_index": round(float(hhi), 4),
            "effective_n": round(float(1 / hhi), 1) if hhi > 0 else 0,
        }

    def drawdown_check(self, nav: pd.Series) -> Dict[str, float]:
        """Compute current drawdown statistics."""
        roll_max = nav.cummax()
        dd = (nav - roll_max) / roll_max
        current_dd = float(dd.iloc[-1])
        max_dd = float(dd.min())
        return {
            "current_drawdown": current_dd,
            "max_drawdown": max_dd,
            "circuit_breaker_triggered": current_dd < -self.max_dd,
        }

    def correlation_check(
        self,
        weights: pd.Series,
        as_of: pd.Timestamp,
        window: int = 63,
    ) -> pd.DataFrame:
        """Return correlation matrix of held positions."""
        tickers = weights.index.tolist()
        hist = self.returns.loc[:as_of].iloc[-window:][tickers]
        return hist.corr().round(3)

    def marginal_risk_contribution(
        self, weights: pd.Series, as_of: pd.Timestamp
    ) -> pd.Series:
        """
        Marginal risk contribution (MRC) for each holding.
        MRC_i = w_i * (Σw)_i / portfolio_vol
        """
        tickers = weights.index.tolist()
        hist = self.returns.loc[:as_of].iloc[-self.var_window:][tickers]
        cov = hist.cov() * 252
        w = weights.values
        port_vol = np.sqrt(w @ cov.values @ w)
        if port_vol == 0:
            return pd.Series(0.0, index=weights.index)
        mrc = (cov.values @ w) * w / port_vol
        return pd.Series(mrc, index=weights.index).round(6)

    def full_risk_report(
        self,
        weights: pd.Series,
        nav: pd.Series,
        as_of: pd.Timestamp,
    ) -> Dict:
        """Compile all risk metrics into a single report dict."""
        report = {}
        report["concentration"] = self.concentration_check(weights)
        report["drawdown"]      = self.drawdown_check(nav)
        report["var_95"]        = self.portfolio_var(weights, as_of)
        report["cvar_95"]       = self.portfolio_cvar(weights, as_of)
        report["parametric_var_95"] = self.parametric_var(weights, as_of)
        mrc = self.marginal_risk_contribution(weights, as_of)
        report["top_risk_contributors"] = mrc.nlargest(5).to_dict()
        return report
