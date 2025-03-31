"""
Trading Engine for Investment Recommendation System
Provides ML-based trading strategies, execution, and backtesting
"""

from trading_engine.src.strategies.base_strategy import BaseStrategy
from trading_engine.src.strategies.ml_strategy import MLStrategy
from trading_engine.src.execution.executor import TradingExecutor
from trading_engine.src.backtest.backtester import Backtester

__all__ = [
    'BaseStrategy',
    'MLStrategy',
    'TradingExecutor',
    'Backtester'
] 