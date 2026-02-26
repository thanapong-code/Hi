"""
US Stocks Momentum Trading Model – Main Entry Point
====================================================

Quick-start (live data):
    python main.py

Quick-start (simulated data, no internet required):
    python main.py --simulate

This script demonstrates the full pipeline:
  1. Fetch (or simulate) price & volume data
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

import pandas as pd

import config
from momentum_model import (
    DataFetcher,
    MomentumSignals,
    PortfolioConstructor,
    BacktestEngine,
    PerformanceMetrics,
    generate_market_data,
    generate_benchmark,
    fetch_github_sp500,
    fetch_stooq_data,
    fetch_yfinance_data,
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
    p.add_argument("--simulate",    action="store_true",
                   help="Use synthetic data (GBM) instead of Yahoo Finance (useful offline)")
    p.add_argument("--real",        action="store_true",
                   help="Use real S&P 500 data from GitHub (2013-2018, no API key needed)")
    p.add_argument("--stooq",       action="store_true",
                   help="Use real data from Stooq.com (free, no API key, supports 2024→today)")
    p.add_argument("--yfinance",    action="store_true",
                   help="Use real data from Yahoo Finance via yfinance (free, no API key, supports 2024→today)")
    p.add_argument("--n-stocks",    type=int, default=50,
                   help="Number of synthetic stocks to simulate (default: 50)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def load_live_data(args: argparse.Namespace):
    """Download real data from Yahoo Finance."""
    import yfinance as yf

    fetcher = DataFetcher(
        tickers=config.SP500_TICKERS,
        start_date=args.start,
        end_date=args.end,
        min_history_days=config.MOMENTUM_WINDOWS["long"] + config.SKIP_RECENT_DAYS + 30,
    ).fetch()

    logger.info("Fetching benchmark (%s)…", config.BENCHMARK_TICKER)
    spy = yf.download(
        config.BENCHMARK_TICKER,
        start=args.start,
        end=args.end,
        auto_adjust=True,
        progress=False,
    )["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.squeeze()
    benchmark = spy.rename("SPY")

    return fetcher.prices, fetcher.volume, benchmark


# ─────────────────────────────────────────────────────────────────────────────
def load_simulated_data(args: argparse.Namespace):
    """Generate synthetic stock + benchmark data (GBM with market factor)."""
    n = args.n_stocks
    tickers = config.SP500_TICKERS[:n]

    logger.info("Generating synthetic data for %d stocks (%s → %s)…",
                n, args.start, args.end or "today")

    prices, volume = generate_market_data(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        seed=42,
    )
    benchmark = generate_benchmark(prices)
    return prices, volume, benchmark


# ─────────────────────────────────────────────────────────────────────────────
def load_real_data(args: argparse.Namespace):
    """Load real S&P 500 data from GitHub (2013-02-08 → 2018-02-07)."""
    prices, volume, benchmark = fetch_github_sp500(
        tickers=config.SP500_TICKERS,
        min_history_days=config.MOMENTUM_WINDOWS["long"] + config.SKIP_RECENT_DAYS + 30,
    )
    return prices, volume, benchmark


# ─────────────────────────────────────────────────────────────────────────────
def load_stooq_data(args: argparse.Namespace):
    """Fetch real data from Stooq.com for any date range (2024→today supported)."""
    prices, volume, benchmark = fetch_stooq_data(
        tickers=config.SP500_TICKERS,
        start_date=args.start,
        end_date=args.end,
        min_history_days=config.MOMENTUM_WINDOWS["long"] + config.SKIP_RECENT_DAYS + 30,
    )
    return prices, volume, benchmark


# ─────────────────────────────────────────────────────────────────────────────
def load_yfinance_data(args: argparse.Namespace):
    """Fetch real data from Yahoo Finance for any date range (2024→today supported)."""
    prices, volume, benchmark = fetch_yfinance_data(
        tickers=config.SP500_TICKERS,
        start_date=args.start,
        end_date=args.end,
        min_history_days=config.MOMENTUM_WINDOWS["long"] + config.SKIP_RECENT_DAYS + 30,
    )
    return prices, volume, benchmark


# ─────────────────────────────────────────────────────────────────────────────
def run_signal_analysis(prices: pd.DataFrame, volume: pd.DataFrame) -> None:
    """Print a snapshot of momentum signals for the latest date."""
    logger.info("Computing momentum signals…")
    sig = MomentumSignals(
        prices=prices,
        volume=volume,
        signal_weights=config.SIGNAL_WEIGHTS,
        momentum_windows=config.MOMENTUM_WINDOWS,
        skip_recent=config.SKIP_RECENT_DAYS,
        rsi_period=config.RSI_PERIOD,
        macd_fast=config.MACD_FAST,
        macd_slow=config.MACD_SLOW,
        macd_signal_period=config.MACD_SIGNAL,
        ema_fast=config.EMA_FAST,
        ema_slow=config.EMA_SLOW,
        ema_filter=config.EMA_FILTER,
    ).compute()

    n_show = min(20, prices.shape[1])
    print("\n" + "=" * 65)
    print(f"  TOP {n_show} MOMENTUM STOCKS (Latest Date: {prices.index[-1].date()})")
    print("=" * 65)
    snap = sig.signal_snapshot().head(n_show)
    print(snap.to_string())
    print("=" * 65)

    n_bottom = min(10, prices.shape[1])
    print("\n" + "=" * 65)
    print(f"  BOTTOM {n_bottom} MOMENTUM STOCKS (Potential Shorts)")
    print("=" * 65)
    print(sig.signal_snapshot().tail(n_bottom).to_string())
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    benchmark: pd.Series,
    args: argparse.Namespace,
) -> None:
    """Run the full walk-forward backtest and report results."""

    engine = BacktestEngine(
        prices=prices,
        volume=volume,
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
        ema_filter=config.EMA_FILTER,
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
        print(f"Total trades:              {len(trade_df)}")
        print(f"Total transaction costs:   ${total_cost:,.0f}")
        print(f"Avg cost per trade:        ${total_cost / len(trade_df):,.2f}\n")

    # Charts
    if args.plot or args.save_charts:
        import matplotlib
        if not args.plot:
            matplotlib.use("Agg")    # headless when only saving
        import matplotlib.pyplot as plt

        save_dir = Path(args.save_charts) if args.save_charts else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        perf_chart_path   = str(save_dir / "performance.png")   if save_dir else None
        weights_chart_path = str(save_dir / "weights.png")       if save_dir else None

        perf.plot(save_path=perf_chart_path)
        perf.plot_weights(result.weights_history, save_path=weights_chart_path)

        if args.plot:
            plt.show()
        elif save_dir:
            logger.info("Charts saved to %s", save_dir)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    if args.simulate:
        mode = "SIMULATE"
    elif args.real:
        mode = "REAL (GitHub SP500)"
    elif args.stooq:
        mode = "REAL (Stooq.com)"
    elif args.yfinance:
        mode = "REAL (Yahoo Finance/yfinance)"
    else:
        mode = "LIVE (Yahoo Finance)"
    logger.info("=== US Stocks Momentum Model ===")
    logger.info("Mode: %s | %s → %s | Rebalance: %s | Top-N: %d | Method: %s",
                mode, args.start, args.end or "today",
                args.rebalance, args.top_n, args.method)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    if args.simulate:
        prices, volume, benchmark = load_simulated_data(args)
    elif args.real:
        prices, volume, benchmark = load_real_data(args)
    elif args.stooq:
        prices, volume, benchmark = load_stooq_data(args)
    elif args.yfinance:
        prices, volume, benchmark = load_yfinance_data(args)
    else:
        prices, volume, benchmark = load_live_data(args)

    logger.info("Data ready: %d stocks × %d trading days.",
                prices.shape[1], prices.shape[0])

    # Quick data summary
    returns = prices.pct_change()
    ann_ret = (1 + returns).prod() ** (252 / returns.count()) - 1
    ann_vol = returns.std() * (252 ** 0.5)
    sharpe  = ann_ret / ann_vol
    summary = pd.DataFrame({"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe})
    print("\nTop 10 stocks by Sharpe ratio:")
    print(summary.nlargest(10, "sharpe").round(3).to_string())

    # ── Step 2: Signal analysis ───────────────────────────────────────────────
    run_signal_analysis(prices, volume)

    if args.signal_only:
        logger.info("--signal-only flag set; skipping backtest.")
        return

    # ── Step 3: Backtest ──────────────────────────────────────────────────────
    run_backtest(prices, volume, benchmark, args)


if __name__ == "__main__":
    main()
