"""
Performance metrics and visualisation for the momentum model.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Compute and display performance statistics for a strategy.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns.
    benchmark_returns : pd.Series
        Daily benchmark returns (e.g. SPY).
    risk_free_rate : float
        Annualised risk-free rate (e.g. 0.045 for 4.5%).
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.045,
    ) -> None:
        self.ret = returns.dropna()
        self.bm_ret = benchmark_returns.dropna()
        self.rf = risk_free_rate
        self._daily_rf = (1 + self.rf) ** (1 / self.TRADING_DAYS) - 1

    # ── Core statistics ───────────────────────────────────────────────────────

    def total_return(self) -> float:
        return float((1 + self.ret).prod() - 1)

    def annualised_return(self) -> float:
        n = len(self.ret)
        return float((1 + self.total_return()) ** (self.TRADING_DAYS / n) - 1)

    def annualised_volatility(self) -> float:
        return float(self.ret.std() * np.sqrt(self.TRADING_DAYS))

    def sharpe_ratio(self) -> float:
        excess = self.ret - self._daily_rf
        if excess.std() == 0:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(self.TRADING_DAYS))

    def sortino_ratio(self) -> float:
        excess = self.ret - self._daily_rf
        downside = excess[excess < 0].std()
        if downside == 0:
            return 0.0
        return float(excess.mean() / downside * np.sqrt(self.TRADING_DAYS))

    def calmar_ratio(self) -> float:
        mdd = self.max_drawdown()
        if mdd == 0:
            return 0.0
        return float(self.annualised_return() / abs(mdd))

    def max_drawdown(self) -> float:
        cum = (1 + self.ret).cumprod()
        rolling_max = cum.cummax()
        dd = (cum - rolling_max) / rolling_max
        return float(dd.min())

    def drawdown_series(self) -> pd.Series:
        cum = (1 + self.ret).cumprod()
        rolling_max = cum.cummax()
        return (cum - rolling_max) / rolling_max

    def value_at_risk(self, confidence: float = 0.95) -> float:
        return float(np.percentile(self.ret, (1 - confidence) * 100))

    def conditional_var(self, confidence: float = 0.95) -> float:
        var = self.value_at_risk(confidence)
        return float(self.ret[self.ret <= var].mean())

    def omega_ratio(self, threshold: float = 0.0) -> float:
        gains = self.ret[self.ret > threshold] - threshold
        losses = threshold - self.ret[self.ret <= threshold]
        if losses.sum() == 0:
            return np.inf
        return float(gains.sum() / losses.sum())

    def win_rate(self) -> float:
        return float((self.ret > 0).mean())

    def profit_factor(self) -> float:
        gains = self.ret[self.ret > 0].sum()
        losses = abs(self.ret[self.ret < 0].sum())
        if losses == 0:
            return np.inf
        return float(gains / losses)

    # ── Benchmark-relative stats ──────────────────────────────────────────────

    def _align(self) -> Tuple[pd.Series, pd.Series]:
        idx = self.ret.index.intersection(self.bm_ret.index)
        return self.ret.loc[idx], self.bm_ret.loc[idx]

    def alpha_beta(self) -> Tuple[float, float]:
        """Jensen's alpha and market beta (annualised alpha)."""
        r, b = self._align()
        if len(r) < 5:
            return 0.0, 1.0
        beta = np.cov(r, b)[0, 1] / np.var(b)
        alpha_daily = r.mean() - beta * b.mean()
        alpha_ann = alpha_daily * self.TRADING_DAYS
        return float(alpha_ann), float(beta)

    def information_ratio(self) -> float:
        r, b = self._align()
        active = r - b
        if active.std() == 0:
            return 0.0
        return float(active.mean() / active.std() * np.sqrt(self.TRADING_DAYS))

    def tracking_error(self) -> float:
        r, b = self._align()
        return float((r - b).std() * np.sqrt(self.TRADING_DAYS))

    def benchmark_total_return(self) -> float:
        return float((1 + self.bm_ret).prod() - 1)

    def benchmark_annualised_return(self) -> float:
        n = len(self.bm_ret)
        return float((1 + self.benchmark_total_return()) ** (self.TRADING_DAYS / n) - 1)

    # ── Summary tables ────────────────────────────────────────────────────────

    def summary(self) -> pd.Series:
        """Return a Series of all key metrics."""
        alpha, beta = self.alpha_beta()
        return pd.Series({
            "Total Return":             f"{self.total_return():.2%}",
            "Annualised Return":        f"{self.annualised_return():.2%}",
            "Annualised Volatility":    f"{self.annualised_volatility():.2%}",
            "Sharpe Ratio":             f"{self.sharpe_ratio():.2f}",
            "Sortino Ratio":            f"{self.sortino_ratio():.2f}",
            "Calmar Ratio":             f"{self.calmar_ratio():.2f}",
            "Max Drawdown":             f"{self.max_drawdown():.2%}",
            "VaR (95%)":                f"{self.value_at_risk():.2%}",
            "CVaR (95%)":               f"{self.conditional_var():.2%}",
            "Omega Ratio":              f"{self.omega_ratio():.2f}",
            "Win Rate":                 f"{self.win_rate():.2%}",
            "Profit Factor":            f"{self.profit_factor():.2f}",
            "Alpha (ann.)":             f"{alpha:.2%}",
            "Beta":                     f"{beta:.2f}",
            "Information Ratio":        f"{self.information_ratio():.2f}",
            "Tracking Error":           f"{self.tracking_error():.2%}",
            "Benchmark Total Return":   f"{self.benchmark_total_return():.2%}",
            "Benchmark Ann. Return":    f"{self.benchmark_annualised_return():.2%}",
        }, name="Strategy")

    def print_summary(self) -> None:
        s = self.summary()
        width = 42
        print("\n" + "=" * width)
        print(" MOMENTUM MODEL – PERFORMANCE SUMMARY")
        print("=" * width)
        for k, v in s.items():
            print(f"  {k:<28} {v:>10}")
        print("=" * width + "\n")

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot(
        self,
        nav: Optional[pd.Series] = None,
        bench_nav: Optional[pd.Series] = None,
        weights_history: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate a 4-panel performance dashboard:
          1. Cumulative NAV vs benchmark
          2. Drawdown
          3. Rolling Sharpe (63-day)
          4. Top holdings heatmap (or monthly returns)
        """
        sns.set_theme(style="darkgrid", palette="muted")
        fig = plt.figure(figsize=(16, 14))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        # ── Panel 1: Cumulative returns ───────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        cum_ret = (1 + self.ret).cumprod()
        cum_bm  = (1 + self.bm_ret).cumprod()
        ax1.plot(cum_ret.index, cum_ret.values, label="Momentum Strategy",
                 color="#2196F3", linewidth=1.8)
        ax1.plot(cum_bm.index, cum_bm.values, label="Benchmark (SPY)",
                 color="#FF9800", linewidth=1.4, linestyle="--", alpha=0.8)
        ax1.set_title("Cumulative Returns", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Growth of $1")
        ax1.legend(loc="upper left")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.1f}"))

        # ── Panel 2: Drawdown ─────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        dd = self.drawdown_series()
        ax2.fill_between(dd.index, dd.values * 100, 0,
                         color="#F44336", alpha=0.6, label="Strategy DD")
        bm_cum = (1 + self.bm_ret).cumprod()
        bm_dd = (bm_cum - bm_cum.cummax()) / bm_cum.cummax()
        ax2.plot(bm_dd.index, bm_dd.values * 100,
                 color="#FF9800", linewidth=1.0, alpha=0.7, label="Benchmark DD")
        ax2.set_title("Drawdown", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)")
        ax2.legend(fontsize=8)

        # ── Panel 3: Rolling Sharpe ───────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        roll_mean = self.ret.rolling(63).mean()
        roll_std  = self.ret.rolling(63).std()
        roll_sharpe = (roll_mean / roll_std) * np.sqrt(self.TRADING_DAYS)
        ax3.plot(roll_sharpe.index, roll_sharpe.values, color="#4CAF50", linewidth=1.4)
        ax3.axhline(0, color="white", linewidth=0.8, linestyle="--")
        ax3.axhline(1, color="#8BC34A", linewidth=0.8, linestyle=":", alpha=0.7)
        ax3.set_title("Rolling 63-day Sharpe Ratio", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Sharpe Ratio")

        # ── Panel 4: Monthly returns heatmap ──────────────────────────────────
        ax4 = fig.add_subplot(gs[2, :])
        monthly = (1 + self.ret).resample("ME").prod() - 1
        monthly_df = monthly.to_frame("return")
        monthly_df["year"]  = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month
        pivot = monthly_df.pivot(index="year", columns="month", values="return")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][: pivot.columns.max()]
        sns.heatmap(
            pivot * 100,
            ax=ax4,
            cmap="RdYlGn",
            center=0,
            fmt=".1f",
            annot=True,
            linewidths=0.5,
            cbar_kws={"label": "Return (%)"},
        )
        ax4.set_title("Monthly Returns Heatmap (%)", fontsize=12, fontweight="bold")
        ax4.set_ylabel("Year")
        ax4.set_xlabel("")

        fig.suptitle("US Stocks Momentum Strategy – Performance Dashboard",
                     fontsize=15, fontweight="bold", y=1.01)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("Chart saved to %s", save_path)

        return fig

    def plot_weights(
        self,
        weights_history: pd.DataFrame,
        top_n: int = 10,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Stacked area chart of top-N holdings over time."""
        # Pick tickers with highest average weight
        avg_w = weights_history.mean().nlargest(top_n)
        df = weights_history[avg_w.index]

        fig, ax = plt.subplots(figsize=(14, 5))
        df.plot.area(ax=ax, colormap="tab20", alpha=0.8, linewidth=0)
        ax.set_title(f"Portfolio Holdings Over Time (Top {top_n})",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Portfolio Weight")
        ax.legend(loc="upper left", fontsize=7, ncol=2)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
        return fig

    def annual_returns_table(self) -> pd.DataFrame:
        """Year-by-year return comparison."""
        annual_strategy = (1 + self.ret).resample("YE").prod() - 1
        annual_bm = (1 + self.bm_ret).resample("YE").prod() - 1
        tbl = pd.DataFrame({
            "Strategy": annual_strategy,
            "Benchmark": annual_bm,
        })
        tbl.index = tbl.index.year
        tbl["Excess"] = tbl["Strategy"] - tbl["Benchmark"]
        return tbl.applymap(lambda x: f"{x:.2%}")
