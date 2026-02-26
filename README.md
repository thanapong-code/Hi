# US Stocks Momentum Trading Model

A quantitative momentum trading system for US equities with multi-factor signals, risk-parity portfolio construction, and walk-forward backtesting.

## Features

| Component | Description |
|-----------|-------------|
| **Multi-factor signals** | Price momentum (1/3/12-month), RSI, MACD, OBV volume trend, inverse-vol adjusted return |
| **Portfolio construction** | Equal-weight, score-weight, risk-parity (inverse-vol), min-variance (QP) |
| **Risk management** | Hard stop-loss, trailing stop, portfolio drawdown circuit-breaker, VaR/CVaR |
| **Backtesting** | Walk-forward engine with realistic costs (transaction fees + slippage), no look-ahead |
| **Performance analytics** | Sharpe, Sortino, Calmar, alpha/beta, Information Ratio, monthly heatmap |

## Architecture

```
momentum_model/
├── data.py        – DataFetcher: download & clean OHLCV data (Yahoo Finance)
├── signals.py     – MomentumSignals: compute & combine sub-signals
├── portfolio.py   – PortfolioConstructor: target weight allocation
├── backtest.py    – BacktestEngine: walk-forward simulation
├── metrics.py     – PerformanceMetrics: stats + visualisations
└── risk.py        – StopLossManager + PortfolioRiskMonitor
config.py          – All tunable parameters
main.py            – CLI entry point
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full backtest with default settings (monthly rebalance, top-20 stocks)
python main.py

# Only compute signals (no backtest)
python main.py --signal-only

# Weekly rebalance, top-30 stocks, equal-weight, save charts
python main.py --rebalance weekly --top-n 30 --method equal --save-charts ./charts

# Quarterly rebalance, min-variance portfolio
python main.py --rebalance quarterly --method min_variance --plot
```

## CLI Options

```
--start         Start date (default: 2015-01-01)
--end           End date (default: today)
--top-n         Number of long positions (default: 20)
--rebalance     weekly | monthly | quarterly (default: monthly)
--method        equal | score_weight | risk_parity | min_variance (default: risk_parity)
--capital       Initial capital in USD (default: 1,000,000)
--plot          Show interactive charts
--save-charts   Directory path to save PNG charts
--signal-only   Skip backtest, only print signal snapshot
```

## Momentum Signals

### 1. Price Momentum (40% weight)
Blended return signal across three horizons, each volatility-adjusted:
- **Long (12-1 month)**: captures trend persistence, skips recent reversal window
- **Medium (3-month)**: intermediate trend
- **Short (1-month)**: near-term momentum

### 2. RSI Signal (15% weight)
RSI values 50–70 indicate rising momentum. Values > 80 or < 20 are penalised (reversal risk).

### 3. MACD Signal (15% weight)
MACD histogram normalised by its rolling standard deviation. Positive histogram = bullish cross.

### 4. Volume Signal (15% weight)
On-Balance Volume (OBV) momentum — measures whether volume is accumulating (positive) or distributing (negative).

### 5. Volatility-Adjusted Momentum (15% weight)
Medium-term return divided by realised volatility. Rewards consistent trends over erratic movers.

## Portfolio Construction Methods

| Method | Description |
|--------|-------------|
| `equal` | 1/N equal weight |
| `score_weight` | Weight proportional to composite momentum score |
| `risk_parity` | Inverse-volatility weighting (each stock contributes equal vol) |
| `min_variance` | Minimum-variance via quadratic programming (Ledoit-Wolf covariance) |

All methods apply position constraints (`MIN_POSITION_SIZE`, `MAX_POSITION_SIZE`) and optional volatility targeting.

## Risk Management

- **Hard stop-loss**: Exit position if it falls >10% from entry price
- **Trailing stop**: Exit if price falls >8% from its peak since entry
- **Drawdown circuit-breaker**: Halt all trading if portfolio drawdown exceeds 20%
- **VaR / CVaR**: Historical and parametric risk measurement at 95% confidence
- **Concentration limits**: Max 10% per name, HHI monitoring

## Configuration

All parameters are in `config.py`:

```python
TOP_N_STOCKS = 20                # long positions
REBALANCE_FREQUENCY = "monthly"
MAX_POSITION_SIZE = 0.10         # 10% per stock
STOP_LOSS_PCT = 0.10             # 10% hard stop
TRAILING_STOP_PCT = 0.08         # 8% trailing stop
MAX_DRAWDOWN_LIMIT = 0.20        # 20% circuit-breaker
TRANSACTION_COST_BPS = 10        # 10 bps per trade
SLIPPAGE_BPS = 5                 # 5 bps slippage
RISK_FREE_RATE = 0.045           # 4.5%
```

## Programmatic Usage

```python
from momentum_model import DataFetcher, MomentumSignals, BacktestEngine, PerformanceMetrics
import config

# 1. Fetch data
fetcher = DataFetcher(
    tickers=config.SP500_TICKERS,
    start_date="2018-01-01",
).fetch()

# 2. Get current momentum signals
signals = MomentumSignals(
    prices=fetcher.prices,
    volume=fetcher.volume,
).compute()

top20 = signals.top_n(20)
print("Top 20 momentum stocks:", top20.tolist())
print(signals.signal_snapshot().head(20))

# 3. Run backtest
import yfinance as yf
spy = yf.download("SPY", start="2018-01-01", auto_adjust=True, progress=False)["Close"].squeeze()

engine = BacktestEngine(
    prices=fetcher.prices,
    volume=fetcher.volume,
    benchmark_prices=spy,
    top_n=20,
    rebalance_freq="monthly",
    weighting_method="risk_parity",
)
result = engine.run()

# 4. Analyse performance
perf = PerformanceMetrics(result.returns, result.benchmark_returns)
perf.print_summary()
fig = perf.plot()
```

## Disclaimer

This model is for **educational and research purposes only**. It is not financial advice. Past backtest performance does not guarantee future results. Always apply your own due diligence before making investment decisions.
