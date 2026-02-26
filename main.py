"""
US Stocks Momentum Trading Model – Main Entry Point
====================================================

Quick-start
-----------
    python main.py

This script demonstrates the full pipeline:
  1. Fetch price & volume data from Yahoo Finance
  2. Compute multi-factor momentum signals
  3. Build a risk-parity weighted portfolio
  4. Run a walk-forward backtest (monthly rebalance)
  5. Print performance statistics and generate charts
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yfinance as yf
import pandas as pd

import config
from momentum_model import (
    DataFetcher,
    MomentumSignals,
    PortfolioConstructor,
    BacktestEngine,
    PerformanceMetrics,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="US Stocks Momentum Model")
    p.add_argument("--start",       default=config.DEFAULT_START_DATE, help="Start date YYYY-MM-DD")
    p.add_argument("--end",         default=config.DEFAULT_END_DATE,   help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--top-n",       type=int, default=config.TOP_N_STOCKS, help="Number of long positions")
    p.add_argument("--rebalance",   default=config.REBALANCE_FREQUENCY,
                   choices=["weekly", "monthly", "quarterly"], help="Rebalance frequency")
    p.add_argument("--method",      default="risk_parity",
                   choices=["equal", "score_weight", "risk_parity", "min_variance"],
                   help="Portfolio weighting method")
    p.add_argument("--capital",     type=float, default=config.INITIAL_CAPITAL, help="Initial capital (USD)")
    p.add_argument("--plot",        action="store_true", help="Show performance charts")
    p.add_argument("--save-charts", default="", help="Directory to save charts (optional)")
    p.add_argument("--signal-only", action="store_true", help="Only compute signals (no backtest)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def fetch_benchmark(start: str, end: str | None) -> pd.Series:
    """Download SPY as benchmark."""
    logger.info("Fetching benchmark (%s)…", config.BENCHMARK_TICKER)
    spy = yf.download(
        config.BENCHMARK_TICKER,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.squeeze()
    return spy.rename("SPY")


# ─────────────────────────────────────────────────────────────────────────────
def run_signal_analysis(fetcher: DataFetcher) -> None:
    """Print a snapshot of momentum signals for the latest date."""
    logger.info("Computing momentum signals…")
    sig = MomentumSignals(
        prices=fetcher.prices,
        volume=fetcher.volume,
        signal_weights=config.SIGNAL_WEIGHTS,
        momentum_windows=config.MOMENTUM_WINDOWS,
        skip_recent=config.SKIP_RECENT_DAYS,
        rsi_period=config.RSI_PERIOD,
        macd_fast=config.MACD_FAST,
        macd_slow=config.MACD_SLOW,
        macd_signal_period=config.MACD_SIGNAL,
    ).compute()

    print("\n" + "=" * 60)
    print("  TOP 20 MOMENTUM STOCKS (Latest Date)")
    print("=" * 60)
    snap = sig.signal_snapshot().head(20)
    print(snap.to_string())
    print("=" * 60)

    print("\n" + "=" * 60)
    print("  BOTTOM 10 MOMENTUM STOCKS (Potential Shorts)")
    print("=" * 60)
    print(sig.signal_snapshot().tail(10).to_string())
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(
    fetcher: DataFetcher,
    benchmark: pd.Series,
    args: argparse.Namespace,
) -> None:
    """Run the full walk-forward backtest and report results."""

    engine = BacktestEngine(
        prices=fetcher.prices,
        volume=fetcher.volume,
        benchmark_prices=benchmark,
        initial_capital=args.capital,
        top_n=args.top_n,
        rebalance_freq=args.rebalance,
        weighting_method=args.method,
        transaction_cost_bps=config.TRANSACTION_COST_BPS,
        slippage_bps=config.SLIPPAGE_BPS,
        stop_loss_pct=config.STOP_LOSS_PCT,
        trailing_stop_pct=config.TRAILING_STOP_PCT,
        max_drawdown_limit=config.MAX_DRAWDOWN_LIMIT,
        max_position=config.MAX_POSITION_SIZE,
        min_position=config.MIN_POSITION_SIZE,
        vol_target=config.MAX_PORTFOLIO_VOLATILITY,
        skip_recent=config.SKIP_RECENT_DAYS,
        signal_weights=config.SIGNAL_WEIGHTS,
        momentum_windows=config.MOMENTUM_WINDOWS,
    )

    result = engine.run()

    # Performance metrics
    perf = PerformanceMetrics(
        returns=result.returns,
        benchmark_returns=result.benchmark_returns,
        risk_free_rate=config.RISK_FREE_RATE,
    )
    perf.print_summary()

    # Annual returns table
    print("Annual Returns:")
    print(perf.annual_returns_table().to_string())
    print()

    # Trade summary
    if result.trades:
        trade_df = pd.DataFrame([
            {
                "date": t.date, "ticker": t.ticker, "direction": t.direction,
                "shares": round(t.shares, 1), "price": round(t.price, 2),
                "value": round(t.value, 0), "cost": round(t.cost, 2),
            }
            for t in result.trades
        ])
        total_cost = trade_df["cost"].sum()
        print(f"Total trades: {len(trade_df)}")
        print(f"Total transaction costs: ${total_cost:,.0f}")
        print(f"Avg cost per trade: ${total_cost / len(trade_df):,.2f}\n")

    # Charts
    if args.plot or args.save_charts:
        save_dir = Path(args.save_charts) if args.save_charts else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        perf_chart_path = str(save_dir / "performance.png") if save_dir else None
        weights_chart_path = str(save_dir / "weights.png") if save_dir else None

        fig1 = perf.plot(save_path=perf_chart_path)
        fig2 = perf.plot_weights(result.weights_history, save_path=weights_chart_path)

        if args.plot:
            import matplotlib.pyplot as plt
            plt.show()


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Step 1: Fetch data ────────────────────────────────────────────────────
    logger.info("=== US Stocks Momentum Model ===")
    logger.info("Universe: %d tickers | %s → %s | Rebalance: %s | Top-N: %d",
                len(config.SP500_TICKERS), args.start, args.end or "today",
                args.rebalance, args.top_n)

    fetcher = DataFetcher(
        tickers=config.SP500_TICKERS,
        start_date=args.start,
        end_date=args.end,
        min_history_days=config.MOMENTUM_WINDOWS["long"] + config.SKIP_RECENT_DAYS + 30,
    ).fetch()

    logger.info("Universe loaded: %d tickers available.", len(fetcher.tickers_available))

    # Print data summary
    summary = fetcher.summary()
    print("\nTop 10 tickers by Sharpe ratio:")
    print(summary.nlargest(10, "sharpe")[["last_price", "annual_return",
                                           "annual_vol", "sharpe"]].to_string())

    # ── Step 2: Signal analysis ───────────────────────────────────────────────
    run_signal_analysis(fetcher)

    if args.signal_only:
        logger.info("--signal-only flag set; skipping backtest.")
        return

    # ── Step 3: Backtest ──────────────────────────────────────────────────────
    benchmark = fetch_benchmark(args.start, args.end)
    run_backtest(fetcher, benchmark, args)


if __name__ == "__main__":
    main()
