"""
Configuration settings for the US Stocks Momentum Trading Model.
"""

# ─── Universe ────────────────────────────────────────────────────────────────
# S&P 500 representative tickers (large-cap US equities)
SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
    "UNH", "LLY", "JPM", "V", "XOM", "AVGO", "PG", "MA", "HD", "COST",
    "MRK", "CVX", "ABBV", "KO", "PEP", "ADBE", "WMT", "BAC", "MCD",
    "CRM", "TMO", "CSCO", "ACN", "ABT", "NKE", "LIN", "DHR", "TXN",
    "VZ", "CMCSA", "NEE", "PM", "RTX", "HON", "UPS", "LOW", "ORCL",
    "QCOM", "IBM", "AMGN", "INTU", "SPGI", "GS", "MS", "BLK", "AXP",
    "DE", "CAT", "BA", "MMM", "GE", "F", "GM", "INTC", "AMD", "MU",
    "AMAT", "LRCX", "KLAC", "MRVL", "ON", "PANW", "CRWD", "SNOW",
    "NOW", "ZM", "UBER", "LYFT", "ABNB", "DASH", "COIN", "HOOD",
    "JPM", "WFC", "C", "USB", "PNC", "TFC", "COF", "AIG", "MET",
    "CVS", "WBA", "UNH", "HUM", "CI", "ANTM", "MOH", "CNC",
    "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "BKR", "MPC", "PSX",
]
SP500_TICKERS = list(dict.fromkeys(SP500_TICKERS))  # deduplicate

# ─── Data ────────────────────────────────────────────────────────────────────
DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = None          # None → today
PRICE_COLUMN = "Adj Close"

# ─── Momentum Signal Parameters ──────────────────────────────────────────────
# Lookback windows (trading days)
MOMENTUM_WINDOWS = {
    "short":  21,    # ~1 month
    "medium": 63,    # ~3 months
    "long":  252,    # ~12 months
}
SKIP_RECENT_DAYS = 21            # skip most-recent month (reversal avoidance)

RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

VOLUME_MOMENTUM_WINDOW = 20      # days for volume trend signal
EARNINGS_SURPRISE_WEIGHT = 0.2   # weight for earnings momentum

# ─── Signal Weighting ────────────────────────────────────────────────────────
SIGNAL_WEIGHTS = {
    "price_momentum": 0.40,
    "rsi_signal":     0.15,
    "macd_signal":    0.15,
    "volume_signal":  0.15,
    "volatility_adj": 0.15,
}

# ─── Portfolio Construction ───────────────────────────────────────────────────
TOP_N_STOCKS = 20                # number of stocks to hold long
BOTTOM_N_STOCKS = 0              # number of stocks to short (0 = long only)
REBALANCE_FREQUENCY = "monthly"  # "weekly" | "monthly" | "quarterly"
MAX_POSITION_SIZE = 0.10         # max 10% per stock
MIN_POSITION_SIZE = 0.01         # min 1% per stock

# ─── Risk Management ─────────────────────────────────────────────────────────
MAX_PORTFOLIO_VOLATILITY = 0.20  # annualised vol target (20%)
STOP_LOSS_PCT = 0.10             # exit position if down 10% from entry
TRAILING_STOP_PCT = 0.08         # trailing stop (8%)
MAX_DRAWDOWN_LIMIT = 0.20        # halt trading if drawdown exceeds 20%

# ─── Benchmark ───────────────────────────────────────────────────────────────
BENCHMARK_TICKER = "SPY"
RISK_FREE_RATE = 0.045           # annualised (4.5%)

# ─── Backtesting ─────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000      # USD
TRANSACTION_COST_BPS = 10        # basis points per trade (one-way)
SLIPPAGE_BPS = 5                 # basis points slippage per trade
