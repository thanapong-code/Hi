"""
US Stocks Momentum Trading Model
=================================
A quantitative momentum trading system for US equities featuring:
  - Multi-factor momentum signals (price, RSI, MACD, volume, volatility)
  - Portfolio construction with risk-parity weighting
  - Walk-forward backtesting engine
  - Comprehensive performance analytics
"""

from .data import DataFetcher
from .signals import MomentumSignals
from .portfolio import PortfolioConstructor
from .backtest import BacktestEngine
from .metrics import PerformanceMetrics
from .risk import StopLossManager, PortfolioRiskMonitor

__version__ = "1.0.0"
__all__ = [
    "DataFetcher",
    "MomentumSignals",
    "PortfolioConstructor",
    "BacktestEngine",
    "PerformanceMetrics",
    "StopLossManager",
    "PortfolioRiskMonitor",
]
