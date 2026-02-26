"""
Portfolio construction for the momentum model.

Supports three weighting schemes:
  - equal        : equal-weight the top-N momentum stocks
  - score_weight : weight proportional to composite momentum score
  - risk_parity  : inverse-volatility weighting (risk-parity)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

WeightingMethod = Literal["equal", "score_weight", "risk_parity", "min_variance"]


class PortfolioConstructor:
    """
    Builds target portfolio weights from momentum signals.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close price history.
    returns : pd.DataFrame
        Daily return history.
    top_n : int
        Number of long positions.
    method : WeightingMethod
        How to allocate within the selected universe.
    max_position : float
        Maximum weight per stock (0–1).
    min_position : float
        Minimum weight per stock for included names (0–1).
    vol_target : float | None
        Annualised portfolio volatility target; if set, the
        portfolio is levered/de-levered to meet this target.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        top_n: int = 20,
        method: WeightingMethod = "risk_parity",
        max_position: float = 0.10,
        min_position: float = 0.01,
        vol_target: Optional[float] = 0.15,
        cov_lookback: int = 63,
    ) -> None:
        self.prices = prices
        self.returns = returns
        self.top_n = top_n
        self.method = method
        self.max_position = max_position
        self.min_position = min_position
        self.vol_target = vol_target
        self.cov_lookback = cov_lookback

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        selected: pd.Index,
        scores: pd.Series,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """
        Compute target weights for `selected` tickers.

        Parameters
        ----------
        selected : pd.Index
            Tickers chosen by the signal model.
        scores : pd.Series
            Composite momentum scores (index = tickers).
        as_of : pd.Timestamp
            Date to use for covariance estimation.

        Returns
        -------
        pd.Series
            Target weights indexed by ticker (sum ≈ 1.0).
        """
        selected = pd.Index([t for t in selected if t in self.prices.columns])
        if len(selected) == 0:
            return pd.Series(dtype=float)

        if self.method == "equal":
            weights = self._equal(selected)
        elif self.method == "score_weight":
            weights = self._score_weight(selected, scores)
        elif self.method == "risk_parity":
            weights = self._risk_parity(selected, as_of)
        elif self.method == "min_variance":
            weights = self._min_variance(selected, as_of)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        weights = self._apply_constraints(weights)

        if self.vol_target is not None:
            weights = self._scale_to_vol_target(weights, as_of)

        return weights

    # ── Weighting schemes ─────────────────────────────────────────────────────

    def _equal(self, selected: pd.Index) -> pd.Series:
        n = len(selected)
        return pd.Series(1.0 / n, index=selected)

    def _score_weight(self, selected: pd.Index, scores: pd.Series) -> pd.Series:
        s = scores.reindex(selected).fillna(0)
        # Shift so all weights are positive
        s = s - s.min() + 1e-6
        return s / s.sum()

    def _risk_parity(self, selected: pd.Index, as_of: pd.Timestamp) -> pd.Series:
        """
        Inverse-volatility weighting (simplified risk-parity).
        Each stock contributes equally to total portfolio variance.
        """
        vol = self._estimate_vol(selected, as_of)
        inv_vol = 1.0 / vol.replace(0, np.nan).fillna(vol.mean())
        return inv_vol / inv_vol.sum()

    def _min_variance(self, selected: pd.Index, as_of: pd.Timestamp) -> pd.Series:
        """Minimum-variance portfolio using quadratic programming."""
        cov = self._estimate_cov(selected, as_of).values
        n = len(selected)
        x0 = np.ones(n) / n

        def portfolio_var(w):
            return w @ cov @ w

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(self.min_position, self.max_position)] * n

        result = minimize(
            portfolio_var, x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if result.success:
            return pd.Series(result.x, index=selected)
        logger.warning("Min-variance optimisation did not converge; falling back to risk-parity.")
        return self._risk_parity(selected, as_of)

    # ── Constraints & scaling ─────────────────────────────────────────────────

    def _apply_constraints(self, weights: pd.Series) -> pd.Series:
        """Clip to [min, max] then renormalise to sum to 1."""
        weights = weights.clip(lower=self.min_position, upper=self.max_position)
        return weights / weights.sum()

    def _scale_to_vol_target(self, weights: pd.Series, as_of: pd.Timestamp) -> pd.Series:
        """
        Scale the full portfolio up or down to hit `vol_target`.
        Excess weight goes to cash (implicitly, by keeping sum ≤ 1).
        """
        port_vol = self._portfolio_vol(weights, as_of)
        if port_vol <= 0:
            return weights
        scalar = self.vol_target / port_vol
        # Cap leverage at 1 (long-only, no cash borrowing beyond 100%)
        scalar = min(scalar, 1.0 / weights.sum())
        return weights * scalar

    # ── Estimation utilities ──────────────────────────────────────────────────

    def _hist_returns(self, selected: pd.Index, as_of: pd.Timestamp) -> pd.DataFrame:
        ret = self.returns[selected]
        ret = ret.loc[:as_of].iloc[-self.cov_lookback:]
        return ret

    def _estimate_cov(self, selected: pd.Index, as_of: pd.Timestamp) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage covariance (annualised)."""
        from sklearn.covariance import LedoitWolf
        hist = self._hist_returns(selected, as_of).dropna()
        if len(hist) < 10:
            return pd.DataFrame(np.eye(len(selected)), index=selected, columns=selected)
        lw = LedoitWolf().fit(hist)
        cov = pd.DataFrame(lw.covariance_ * 252, index=selected, columns=selected)
        return cov

    def _estimate_vol(self, selected: pd.Index, as_of: pd.Timestamp) -> pd.Series:
        hist = self._hist_returns(selected, as_of)
        return hist.std() * np.sqrt(252)

    def _portfolio_vol(self, weights: pd.Series, as_of: pd.Timestamp) -> float:
        cov = self._estimate_cov(weights.index, as_of).values
        w = weights.values
        return float(np.sqrt(w @ cov @ w))

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def holdings_table(
        self,
        weights: pd.Series,
        scores: pd.Series,
        prices_row: pd.Series,
        capital: float = 1_000_000,
    ) -> pd.DataFrame:
        """
        Return a human-readable DataFrame of holdings.

        Columns: weight, score, price, market_value, shares
        """
        tbl = pd.DataFrame({"weight": weights, "score": scores.reindex(weights.index)})
        tbl["price"] = prices_row.reindex(weights.index)
        tbl["market_value"] = tbl["weight"] * capital
        tbl["shares"] = (tbl["market_value"] / tbl["price"]).apply(np.floor)
        return tbl.sort_values("weight", ascending=False).round(4)
