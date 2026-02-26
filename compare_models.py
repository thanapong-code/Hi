"""
Model Comparison: Old vs New (EMA 50/200 + RS Rating)
======================================================
Runs two backtests on identical data and prints a side-by-side table.

Old model: price_momentum=0.40, rsi=0.15, macd=0.15, volume=0.15, vol_adj=0.15
New model: adds ema_trend=0.15, rs_rating=0.15; reduces all legacy weights.

Usage:
    python compare_models.py          # uses --real GitHub data (2013-2018)
    python compare_models.py --sim    # uses synthetic GBM data
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

import config
from momentum_model import (
    BacktestEngine,
    PerformanceMetrics,
    generate_market_data,
    generate_benchmark,
    fetch_github_sp500,
)

logging.basicConfig(
    level=logging.WARNING,          # suppress verbose logs during comparison
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Model configs ─────────────────────────────────────────────────────────────

OLD_WEIGHTS = {
    "price_momentum": 0.40,
    "rsi_signal":     0.15,
    "macd_signal":    0.15,
    "volume_signal":  0.15,
    "volatility_adj": 0.15,
}

NEW_WEIGHTS = {
    "price_momentum": 0.30,
    "rsi_signal":     0.10,
    "macd_signal":    0.10,
    "volume_signal":  0.10,
    "volatility_adj": 0.10,
    "ema_trend":      0.15,
    "rs_rating":      0.15,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_backtest(prices, volume, benchmark, signal_weights, label):
    print(f"  Running backtest [{label}]…", end=" ", flush=True)
    engine = BacktestEngine(
        prices=prices,
        volume=volume,
        benchmark_prices=benchmark,
        initial_capital=config.INITIAL_CAPITAL,
        top_n=config.TOP_N_STOCKS,
        rebalance_freq=config.REBALANCE_FREQUENCY,
        weighting_method="risk_parity",
        transaction_cost_bps=config.TRANSACTION_COST_BPS,
        slippage_bps=config.SLIPPAGE_BPS,
        stop_loss_pct=config.STOP_LOSS_PCT,
        trailing_stop_pct=config.TRAILING_STOP_PCT,
        max_drawdown_limit=config.MAX_DRAWDOWN_LIMIT,
        max_position=config.MAX_POSITION_SIZE,
        min_position=config.MIN_POSITION_SIZE,
        vol_target=config.MAX_PORTFOLIO_VOLATILITY,
        skip_recent=config.SKIP_RECENT_DAYS,
        signal_weights=signal_weights,
        momentum_windows=config.MOMENTUM_WINDOWS,
    )
    result = engine.run()
    print("done.")
    return result


def _metrics_dict(result, label):
    perf = PerformanceMetrics(
        returns=result.returns,
        benchmark_returns=result.benchmark_returns,
        risk_free_rate=config.RISK_FREE_RATE,
    )
    alpha, beta = perf.alpha_beta()
    return {
        "label":               label,
        "Total Return":        perf.total_return(),
        "Ann. Return":         perf.annualised_return(),
        "Ann. Volatility":     perf.annualised_volatility(),
        "Sharpe Ratio":        perf.sharpe_ratio(),
        "Sortino Ratio":       perf.sortino_ratio(),
        "Calmar Ratio":        perf.calmar_ratio(),
        "Max Drawdown":        perf.max_drawdown(),
        "VaR (95%)":           perf.value_at_risk(),
        "CVaR (95%)":          perf.conditional_var(),
        "Omega Ratio":         perf.omega_ratio(),
        "Win Rate":            perf.win_rate(),
        "Profit Factor":       perf.profit_factor(),
        "Alpha (ann.)":        alpha,
        "Beta":                beta,
        "Information Ratio":   perf.information_ratio(),
        "Tracking Error":      perf.tracking_error(),
        "Benchmark Ann. Ret":  perf.benchmark_annualised_return(),
    }


def _annual_table(result_old, result_new):
    """Year-by-year returns for both models + benchmark."""
    def _annual(ret):
        return (1 + ret).resample("YE").prod() - 1

    old_ann = _annual(result_old.returns)
    new_ann = _annual(result_new.returns)
    bm_ann  = _annual(result_old.benchmark_returns)

    tbl = pd.DataFrame({
        "Old Model": old_ann,
        "New Model (EMA+RS)": new_ann,
        "Benchmark (SPY)":    bm_ann,
    })
    tbl.index = tbl.index.year
    diff = tbl["New Model (EMA+RS)"] - tbl["Old Model"]
    tbl["Delta (New-Old)"] = diff
    return tbl


def _print_comparison(old: dict, new: dict):
    pct_keys = {
        "Total Return", "Ann. Return", "Ann. Volatility",
        "Max Drawdown", "VaR (95%)", "CVaR (95%)",
        "Alpha (ann.)", "Tracking Error", "Benchmark Ann. Ret", "Win Rate",
    }

    print("\n" + "=" * 75)
    print("  OLD MODEL vs NEW MODEL (EMA 50/200 + RS Rating)")
    print("=" * 75)
    print(f"  {'Metric':<26} {'Old Model':>14}  {'New Model':>14}  {'Delta':>10}")
    print("-" * 75)

    for key in [k for k in old if k != "label"]:
        o_val = old[key]
        n_val = new[key]
        delta = n_val - o_val

        if key in pct_keys:
            o_str = f"{o_val:>13.2%}"
            n_str = f"{n_val:>13.2%}"
            d_str = f"{delta:>+9.2%}" if abs(delta) >= 0.0001 else "       —"
        else:
            o_str = f"{o_val:>13.4f}"
            n_str = f"{n_val:>13.4f}"
            d_str = f"{delta:>+9.4f}" if abs(delta) >= 0.0001 else "       —"

        # Flag improvements (higher Sharpe/return/etc, lower drawdown/vol)
        better_if_higher = {
            "Total Return", "Ann. Return", "Sharpe Ratio", "Sortino Ratio",
            "Calmar Ratio", "Omega Ratio", "Win Rate", "Profit Factor",
            "Alpha (ann.)", "Information Ratio",
        }
        better_if_lower = {"Max Drawdown", "Ann. Volatility", "VaR (95%)", "CVaR (95%)"}

        flag = ""
        if key in better_if_higher and delta > 0.0001:
            flag = " ▲"
        elif key in better_if_lower and delta < -0.0001:
            flag = " ▲"
        elif key in better_if_higher and delta < -0.0001:
            flag = " ▼"
        elif key in better_if_lower and delta > 0.0001:
            flag = " ▼"

        print(f"  {key:<26}{o_str}  {n_str}  {d_str}{flag}")
    print("=" * 75)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare old vs new momentum model")
    parser.add_argument("--sim", action="store_true", help="Use synthetic data instead of real data")
    parser.add_argument("--n-stocks", type=int, default=50, help="Stocks for simulation")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.sim:
        print("Loading synthetic (GBM) data…")
        tickers = config.SP500_TICKERS[:args.n_stocks]
        prices, volume = generate_market_data(tickers=tickers, start_date="2015-01-01", seed=42)
        benchmark = generate_benchmark(prices)
        data_label = "Synthetic GBM 2015–2023"
    else:
        print("Loading real S&P 500 data (GitHub, 2013-2018)…")
        prices, volume, benchmark = fetch_github_sp500(
            tickers=config.SP500_TICKERS,
            min_history_days=config.MOMENTUM_WINDOWS["long"] + config.SKIP_RECENT_DAYS + 30,
        )
        data_label = "Real S&P500 2013–2018 (GitHub)"

    print(f"Data: {prices.shape[1]} stocks × {prices.shape[0]} days  [{data_label}]")

    # ── Run both backtests ────────────────────────────────────────────────────
    result_old = _run_backtest(prices, volume, benchmark, OLD_WEIGHTS, "Old Model")
    result_new = _run_backtest(prices, volume, benchmark, NEW_WEIGHTS, "New Model")

    # ── Comparison table ──────────────────────────────────────────────────────
    old_m = _metrics_dict(result_old, "Old Model")
    new_m = _metrics_dict(result_new, "New Model")
    _print_comparison(old_m, new_m)

    # ── Annual returns ────────────────────────────────────────────────────────
    print("\nAnnual Returns:")
    print("-" * 60)
    ann = _annual_table(result_old, result_new)
    for year, row in ann.iterrows():
        delta_flag = "▲" if row["Delta (New-Old)"] > 0.001 else ("▼" if row["Delta (New-Old)"] < -0.001 else "—")
        print(f"  {year}   Old: {row['Old Model']:>+8.2%}   "
              f"New: {row['New Model (EMA+RS)']:>+8.2%}   "
              f"SPY: {row['Benchmark (SPY)']:>+8.2%}   "
              f"Delta: {row['Delta (New-Old)']:>+7.2%} {delta_flag}")
    print("-" * 60)

    # ── Trade counts ─────────────────────────────────────────────────────────
    print(f"\nTrade count  — Old: {len(result_old.trades)}   New: {len(result_new.trades)}")
    old_cost = sum(t.cost for t in result_old.trades)
    new_cost = sum(t.cost for t in result_new.trades)
    print(f"Total costs  — Old: ${old_cost:,.0f}   New: ${new_cost:,.0f}")
    print()


if __name__ == "__main__":
    main()
