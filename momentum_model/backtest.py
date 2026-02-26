"""
Walk-forward backtesting engine for the momentum model.

Architecture
------------
  BacktestEngine.run()
    └── for each rebalance date:
          1. compute momentum signals (look-back only)
          2. select top-N stocks
          3. build target weights
          4. apply stop-loss / trailing-stop filters
          5. calculate transaction costs
          6. mark-to-market daily until next rebalance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .signals import MomentumSignals
from .portfolio import PortfolioConstructor, WeightingMethod

logger = logging.getLogger(__name__)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    direction: str          # "buy" | "sell"
    shares: float
    price: float
    value: float
    cost_bps: float

    @property
    def cost(self) -> float:
        return abs(self.value) * self.cost_bps / 10_000


@dataclass
class BacktestResult:
    portfolio_values: pd.Series         # daily NAV
    benchmark_values: pd.Series         # daily benchmark NAV
    weights_history: pd.DataFrame       # daily target weights
    trades: List[Trade] = field(default_factory=list)
    rebalance_dates: List[pd.Timestamp] = field(default_factory=list)

    @property
    def returns(self) -> pd.Series:
        return self.portfolio_values.pct_change().dropna()

    @property
    def benchmark_returns(self) -> pd.Series:
        return self.benchmark_values.pct_change().dropna()


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven momentum backtest.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price history for the universe.
    volume : pd.DataFrame
        Volume history (same index/columns as prices).
    benchmark_prices : pd.Series
        Benchmark price series (e.g. SPY).
    initial_capital : float
        Starting portfolio value in USD.
    top_n : int
        Number of long positions.
    rebalance_freq : str
        Pandas offset alias – 'BMS' = monthly, 'W-FRI' = weekly, 'BQS' = quarterly.
    weighting_method : WeightingMethod
        Portfolio weighting scheme.
    transaction_cost_bps : float
        One-way cost in basis points.
    slippage_bps : float
        Additional slippage in basis points.
    stop_loss_pct : float
        Hard stop-loss from entry price (fraction, e.g. 0.10).
    trailing_stop_pct : float
        Trailing stop from peak price since entry (fraction).
    max_drawdown_limit : float
        If portfolio drawdown exceeds this, halt trading.
    signal_weights : dict | None
        Forwarded to MomentumSignals.
    momentum_windows : dict | None
        Forwarded to MomentumSignals.
    """

    FREQ_MAP = {
        "weekly":    "W-FRI",
        "monthly":   "BMS",
        "quarterly": "BQS",
    }

    def __init__(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        benchmark_prices: pd.Series,
        initial_capital: float = 1_000_000,
        top_n: int = 20,
        rebalance_freq: str = "monthly",
        weighting_method: WeightingMethod = "risk_parity",
        transaction_cost_bps: float = 10,
        slippage_bps: float = 5,
        stop_loss_pct: float = 0.10,
        trailing_stop_pct: float = 0.08,
        max_drawdown_limit: float = 0.20,
        max_position: float = 0.10,
        min_position: float = 0.01,
        vol_target: Optional[float] = 0.15,
        skip_recent: int = 21,
        min_history_days: int = 252,
        signal_weights: Optional[Dict[str, float]] = None,
        momentum_windows: Optional[Dict[str, int]] = None,
    ) -> None:
        self.prices = prices
        self.volume = volume
        self.benchmark = benchmark_prices
        self.capital = initial_capital
        self.top_n = top_n
        self.freq = self.FREQ_MAP.get(rebalance_freq, rebalance_freq)
        self.weighting_method = weighting_method
        self.tc_bps = transaction_cost_bps
        self.slip_bps = slippage_bps
        self.stop_loss = stop_loss_pct
        self.trailing_stop = trailing_stop_pct
        self.max_dd = max_drawdown_limit
        self.max_pos = max_position
        self.min_pos = min_position
        self.vol_target = vol_target
        self.skip_recent = skip_recent
        self.min_history = min_history_days
        self.signal_weights = signal_weights
        self.momentum_windows = momentum_windows

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """Execute the full walk-forward backtest."""
        logger.info("Starting backtest…")

        dates = self.prices.index
        rebalance_dates = self._rebalance_dates(dates)

        nav = pd.Series(index=dates, dtype=float)
        bench_nav = pd.Series(index=dates, dtype=float)
        weights_history = pd.DataFrame(0.0, index=dates, columns=self.prices.columns)

        current_weights = pd.Series(dtype=float)
        current_nav = float(self.capital)
        trades: List[Trade] = []
        peak_nav = current_nav

        # Track entry prices and high-water marks for stop-loss
        entry_prices: Dict[str, float] = {}
        hwm_prices: Dict[str, float] = {}

        # Initialise benchmark
        bench_start = self.benchmark.dropna().iloc[0]

        halted = False  # drawdown circuit-breaker

        for i, date in enumerate(dates):
            if date not in self.prices.index:
                continue

            price_row = self.prices.loc[date]

            # ── Daily mark-to-market ──────────────────────────────────────────
            if len(current_weights) > 0:
                port_ret = (current_weights * price_row.reindex(current_weights.index)
                            .pct_change().fillna(0)).sum()
                current_nav *= (1 + port_ret)

            nav[date] = current_nav
            bench_val = (self.benchmark.loc[date] / bench_start) * self.capital
            bench_nav[date] = bench_val

            # Update trailing stop high-water marks
            for ticker in list(current_weights.index):
                if ticker in self.prices.columns:
                    p = price_row.get(ticker, np.nan)
                    if not np.isnan(p):
                        hwm_prices[ticker] = max(hwm_prices.get(ticker, p), p)

            # ── Stop-loss filter (daily check) ────────────────────────────────
            if not halted and len(current_weights) > 0:
                to_remove = []
                for ticker in current_weights.index:
                    p = price_row.get(ticker, np.nan)
                    entry = entry_prices.get(ticker, p)
                    hwm   = hwm_prices.get(ticker, p)
                    if np.isnan(p) or entry == 0 or hwm == 0:
                        continue
                    hard_stop = (p - entry) / entry < -self.stop_loss
                    trail_stop = (p - hwm) / hwm < -self.trailing_stop
                    if hard_stop or trail_stop:
                        reason = "hard stop" if hard_stop else "trailing stop"
                        logger.debug("%s: %s triggered on %s", ticker, reason, date)
                        to_remove.append(ticker)

                if to_remove:
                    exit_wt = current_weights[to_remove].sum()
                    current_weights = current_weights.drop(to_remove)
                    # Renormalise remaining
                    if current_weights.sum() > 0:
                        current_weights /= current_weights.sum()
                    for t in to_remove:
                        entry_prices.pop(t, None)
                        hwm_prices.pop(t, None)

            # ── Drawdown circuit-breaker ──────────────────────────────────────
            peak_nav = max(peak_nav, current_nav)
            drawdown = (current_nav - peak_nav) / peak_nav
            if drawdown < -self.max_dd and not halted:
                logger.warning("Max drawdown %.1f%% breached on %s – halting.", drawdown * 100, date)
                halted = True
                current_weights = pd.Series(dtype=float)

            # Reset halt if we recover to within half the limit
            if halted and drawdown > -self.max_dd / 2:
                logger.info("Drawdown recovered – resuming trading on %s.", date)
                halted = False

            # ── Rebalance ─────────────────────────────────────────────────────
            if date in rebalance_dates and not halted:
                new_weights, new_trades = self._rebalance(
                    date, current_weights, current_nav, entry_prices, hwm_prices
                )
                trades.extend(new_trades)
                current_weights = new_weights

                # Deduct transaction costs from NAV
                total_cost = sum(t.cost for t in new_trades)
                current_nav -= total_cost

            # Record weights
            if len(current_weights) > 0:
                weights_history.loc[date, current_weights.index] = current_weights.values

        result = BacktestResult(
            portfolio_values=nav.dropna(),
            benchmark_values=bench_nav.dropna(),
            weights_history=weights_history,
            trades=trades,
            rebalance_dates=rebalance_dates,
        )
        logger.info(
            "Backtest complete. %d rebalances, %d trades.",
            len(rebalance_dates), len(trades),
        )
        return result

    # ── Rebalance logic ───────────────────────────────────────────────────────

    def _rebalance(
        self,
        date: pd.Timestamp,
        current_weights: pd.Series,
        current_nav: float,
        entry_prices: Dict[str, float],
        hwm_prices: Dict[str, float],
    ) -> Tuple[pd.Series, List[Trade]]:
        """Compute new weights and generate trade objects."""

        # Only use price history up to (not including) today → no look-ahead
        hist_prices = self.prices.loc[:date]
        hist_volume = self.volume.loc[:date]

        # Require minimum history
        valid = hist_prices.count() >= self.min_history
        hist_prices = hist_prices.loc[:, valid]
        hist_volume = hist_volume.loc[:, valid.reindex(hist_volume.columns, fill_value=False)]

        if hist_prices.shape[1] == 0 or hist_prices.shape[0] < self.min_history:
            logger.debug("Insufficient history at %s; skipping rebalance.", date)
            return current_weights, []

        # Signals
        sig = MomentumSignals(
            prices=hist_prices,
            volume=hist_volume,
            signal_weights=self.signal_weights,
            momentum_windows=self.momentum_windows,
            skip_recent=self.skip_recent,
        ).compute()

        selected = sig.top_n(self.top_n, date=date)
        scores   = sig.composite.loc[date]

        # Portfolio weights
        constructor = PortfolioConstructor(
            prices=hist_prices,
            returns=hist_prices.pct_change(),
            top_n=self.top_n,
            method=self.weighting_method,
            max_position=self.max_pos,
            min_position=self.min_pos,
            vol_target=self.vol_target,
        )
        new_weights = constructor.build(selected, scores, date)

        if new_weights.empty:
            return current_weights, []

        # Generate trades (diff between old and new)
        trades = []
        price_row = self.prices.loc[date]
        all_tickers = new_weights.index.union(current_weights.index)

        for ticker in all_tickers:
            old_w = current_weights.get(ticker, 0.0)
            new_w = new_weights.get(ticker, 0.0)
            delta_w = new_w - old_w
            if abs(delta_w) < 1e-6:
                continue
            p = price_row.get(ticker, np.nan)
            if np.isnan(p) or p <= 0:
                continue

            # Apply slippage
            effective_price = p * (1 + np.sign(delta_w) * self.slip_bps / 10_000)
            trade_value = delta_w * current_nav
            shares = trade_value / effective_price
            direction = "buy" if delta_w > 0 else "sell"
            t = Trade(
                date=date,
                ticker=ticker,
                direction=direction,
                shares=shares,
                price=effective_price,
                value=trade_value,
                cost_bps=self.tc_bps,
            )
            trades.append(t)

            # Update entry price tracking
            if direction == "buy" and ticker not in entry_prices:
                entry_prices[ticker] = effective_price
                hwm_prices[ticker]   = effective_price
            elif new_w == 0:
                entry_prices.pop(ticker, None)
                hwm_prices.pop(ticker, None)

        return new_weights, trades

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _rebalance_dates(self, dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """Generate rebalance dates aligned to the trading calendar."""
        ideal = pd.date_range(dates[0], dates[-1], freq=self.freq)
        result = []
        for d in ideal:
            # Find the nearest actual trading day on or after d
            future = dates[dates >= d]
            if len(future):
                result.append(future[0])
        return sorted(set(result))
