"""
3-Way Model Comparison
======================
Runs three backtests on identical data and prints a side-by-side table.

  Model A – Original   : 5 signals, no EMA/RS, no EMA filter
  Model B – Weighted   : 6 signals (adds RS Rating), no EMA filter
  Model C – EMA Filter : 6 signals + RS Rating + EMA-200 hard filter  ← new

Usage:
    python compare_models.py          # real GitHub S&P500 data (2013–2018)
    python compare_models.py --sim    # synthetic GBM data
"""

from __future__ import annotations

import argparse
import logging

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

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")

# ── Model signal-weight configs ───────────────────────────────────────────────

WEIGHTS_OLD = {
    "price_momentum": 0.40,
    "rsi_signal":     0.15,
    "macd_signal":    0.15,
    "volume_signal":  0.15,
    "volatility_adj": 0.15,
}

WEIGHTS_RS = {                    # RS Rating added, no EMA filter
    "price_momentum": 0.35,
    "rsi_signal":     0.10,
    "macd_signal":    0.10,
    "volume_signal":  0.10,
    "volatility_adj": 0.10,
    "rs_rating":      0.25,
}

WEIGHTS_EMA_FILTER = WEIGHTS_RS   # same weights; filter is the extra layer

MODELS = [
    dict(label="A – Original (no EMA/RS)",    weights=WEIGHTS_OLD,        ema_filter=False),
    dict(label="B – +RS Rating (no filter)",  weights=WEIGHTS_RS,         ema_filter=False),
    dict(label="C – +RS + EMA-200 filter",    weights=WEIGHTS_EMA_FILTER, ema_filter=True),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(prices, volume, benchmark, weights, ema_filter, label):
    print(f"  [{label}]…", end=" ", flush=True)
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
        signal_weights=weights,
        momentum_windows=config.MOMENTUM_WINDOWS,
        ema_filter=ema_filter,
    )
    result = engine.run()
    print("done.")
    return result


def _stats(result):
    perf = PerformanceMetrics(
        returns=result.returns,
        benchmark_returns=result.benchmark_returns,
        risk_free_rate=config.RISK_FREE_RATE,
    )
    alpha, beta = perf.alpha_beta()
    return dict(
        total_ret     = perf.total_return(),
        ann_ret       = perf.annualised_return(),
        ann_vol       = perf.annualised_volatility(),
        sharpe        = perf.sharpe_ratio(),
        sortino       = perf.sortino_ratio(),
        calmar        = perf.calmar_ratio(),
        max_dd        = perf.max_drawdown(),
        var95         = perf.value_at_risk(),
        cvar95        = perf.conditional_var(),
        omega         = perf.omega_ratio(),
        win_rate      = perf.win_rate(),
        profit_factor = perf.profit_factor(),
        alpha         = alpha,
        beta          = beta,
        info_ratio    = perf.information_ratio(),
        track_err     = perf.tracking_error(),
        bm_ann_ret    = perf.benchmark_annualised_return(),
        n_trades      = len(result.trades),
        total_cost    = sum(t.cost for t in result.trades),
    )


def _annual(result):
    ann = (1 + result.returns).resample("YE").prod() - 1
    bm  = (1 + result.benchmark_returns).resample("YE").prod() - 1
    return ann, bm


def _print_table(results, labels):
    stats = [_stats(r) for r in results]
    bm_ret = stats[0]["bm_ann_ret"]

    # direction: +1 = higher is better, -1 = lower is better
    ROWS = [
        ("Total Return",       "total_ret",     True,  "{:>9.2%}"),
        ("Ann. Return",        "ann_ret",        True,  "{:>9.2%}"),
        ("Ann. Volatility",    "ann_vol",        False, "{:>9.2%}"),
        ("Sharpe Ratio",       "sharpe",         True,  "{:>9.4f}"),
        ("Sortino Ratio",      "sortino",        True,  "{:>9.4f}"),
        ("Calmar Ratio",       "calmar",         True,  "{:>9.4f}"),
        ("Max Drawdown",       "max_dd",         False, "{:>9.2%}"),
        ("VaR (95%)",          "var95",          False, "{:>9.2%}"),
        ("CVaR (95%)",         "cvar95",         False, "{:>9.2%}"),
        ("Omega Ratio",        "omega",          True,  "{:>9.4f}"),
        ("Win Rate",           "win_rate",       True,  "{:>9.2%}"),
        ("Profit Factor",      "profit_factor",  True,  "{:>9.4f}"),
        ("Alpha (ann.)",       "alpha",          True,  "{:>9.2%}"),
        ("Beta",               "beta",           None,  "{:>9.4f}"),
        ("Information Ratio",  "info_ratio",     True,  "{:>9.4f}"),
        ("Tracking Error",     "track_err",      False, "{:>9.2%}"),
        ("Benchmark Ann. Ret", "bm_ann_ret",     None,  "{:>9.2%}"),
        ("Trades",             "n_trades",       None,  "{:>9d}"),
        ("Total Costs ($)",    "total_cost",     False, "{:>9,.0f}"),
    ]

    col_w = 16
    head  = f"  {'Metric':<24}" + "".join(f"{lb:>{col_w}}" for lb in labels)
    sep   = "=" * (24 + col_w * len(labels) + 2)

    print("\n" + sep)
    print("  3-WAY MODEL COMPARISON")
    print(sep)
    print(head)
    print("-" * len(sep))

    for display, key, higher_better, fmt in ROWS:
        vals = [s[key] for s in stats]
        if key == "n_trades":
            strs = [f"{v:>9d}" for v in vals]
        elif key == "total_cost":
            strs = [f"{v:>9,.0f}" for v in vals]
        else:
            strs = [fmt.format(v) for v in vals]

        # Best value marker
        markers = ["  "] * len(vals)
        if higher_better is True:
            best_idx = max(range(len(vals)), key=lambda i: vals[i])
            markers[best_idx] = " ★"
        elif higher_better is False:
            best_idx = min(range(len(vals)), key=lambda i: vals[i])
            markers[best_idx] = " ★"

        row = f"  {display:<24}" + "".join(
            f"{s:>{col_w - 2}}{markers[i]}" for i, s in enumerate(strs)
        )
        print(row)

    print(sep)
    print("  ★ = best value for that metric")
    print(sep)


def _print_annual(results, labels):
    print("\nAnnual Returns vs Benchmark (SPY)")
    print("-" * 75)
    ann_series = [_annual(r) for r in results]
    all_years  = sorted(set().union(*[s[0].index.year for s in ann_series]))

    hdr = f"  {'Year':<6}" + "".join(f"{lb[:14]:>14}" for lb in labels) + f"  {'SPY':>8}"
    print(hdr)
    print("-" * 75)

    for yr in all_years:
        ts = pd.Timestamp(f"{yr}-12-31")
        row = f"  {yr:<6}"
        bm_val = None
        for i, (ann, bm) in enumerate(ann_series):
            val = ann.get(ts, np.nan)
            if not np.isnan(val):
                row += f"{val:>+13.2%} "
                if bm_val is None:
                    bm_val = bm.get(ts, np.nan)
            else:
                row += f"{'—':>14}"
        if bm_val is not None and not np.isnan(bm_val):
            row += f"  {bm_val:>+7.2%}"
        print(row)
    print("-" * 75)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3-way momentum model comparison")
    parser.add_argument("--sim", action="store_true", help="Use synthetic GBM data")
    parser.add_argument("--n-stocks", type=int, default=50)
    args = parser.parse_args()

    if args.sim:
        print("Loading synthetic (GBM) data…")
        tickers = config.SP500_TICKERS[:args.n_stocks]
        prices, volume = generate_market_data(tickers=tickers, start_date="2015-01-01", seed=42)
        benchmark = generate_benchmark(prices)
        print(f"Data: {prices.shape[1]} stocks × {prices.shape[0]} days  [Synthetic GBM]")
    else:
        print("Loading real S&P500 data (GitHub, 2013–2018)…")
        prices, volume, benchmark = fetch_github_sp500(
            tickers=config.SP500_TICKERS,
            min_history_days=config.MOMENTUM_WINDOWS["long"] + config.SKIP_RECENT_DAYS + 30,
        )
        print(f"Data: {prices.shape[1]} stocks × {prices.shape[0]} days  [Real S&P500 2013–2018]")

    print("\nRunning backtests…")
    results = [_run(prices, volume, benchmark, m["weights"], m["ema_filter"], m["label"])
               for m in MODELS]

    labels = [m["label"] for m in MODELS]
    _print_table(results, labels)
    _print_annual(results, labels)
    print()


if __name__ == "__main__":
    main()
