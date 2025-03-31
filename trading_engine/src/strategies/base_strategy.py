#!/usr/bin/env python3
"""
Base Strategy for Trading Engine
Defines the interface for all trading strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, params: Dict = None):
        """Initialize the strategy with name and params"""
        self.name = name
        self.params = params or {}
        self.position = 0  # Current position: 0 = neutral, 1 = long, -1 = short
        self.last_signal = 0  # Last signal generated
        self.portfolio_value = 1.0  # Normalized portfolio value
        self.trades = []  # Track trade history
    
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> int:
        """
        Generate a trading signal based on the data
        Returns:
          1 for buy signal
          0 for hold/neutral
         -1 for sell signal
        """
        pass
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.Series:
        """Calculate returns based on signals"""
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Generate signals for each data point
        signals = pd.Series(
            [self.generate_signal(data.iloc[:i+1]) for i in range(len(data))],
            index=data.index
        )
        
        # Calculate daily returns
        daily_returns = data['Close'].pct_change().fillna(0)
        
        # Calculate strategy returns (signal * next day's return)
        # Shift signals to implement signals on the next day
        strategy_returns = signals.shift(1) * daily_returns
        strategy_returns.fillna(0, inplace=True)
        
        # Track position changes
        positions = signals.copy()
        self.trades = []
        
        # Record trades
        for i in range(1, len(positions)):
            if positions.iloc[i] != positions.iloc[i-1] and positions.iloc[i] != 0:
                self.trades.append({
                    'date': positions.index[i],
                    'action': 'BUY' if positions.iloc[i] == 1 else 'SELL',
                    'price': data['Close'].iloc[i],
                    'position': positions.iloc[i]
                })
        
        return strategy_returns
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """Run backtest on historical data"""
        # Calculate returns
        strategy_returns = self.calculate_returns(data)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Calculate drawdown
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / rolling_max) - 1
        
        # Calculate performance metrics
        total_return = cumulative_returns.iloc[-1] - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = drawdown.min()
        
        # Buy and hold benchmark
        benchmark_returns = data['Close'].pct_change().fillna(0)
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        benchmark_total_return = benchmark_cumulative.iloc[-1] - 1
        
        # Outperformance
        outperformance = total_return - benchmark_total_return
        
        # Result
        return {
            'strategy_name': self.name,
            'params': self.params,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_return': benchmark_total_return,
            'outperformance': outperformance,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'benchmark_cumulative': benchmark_cumulative,
            'drawdown': drawdown,
            'trades': self.trades,
            'num_trades': len(self.trades)
        }
    
    def get_current_position(self) -> int:
        """Get current position"""
        return self.position
    
    def update_position(self, new_position: int):
        """Update current position"""
        self.position = new_position
    
    def reset(self):
        """Reset strategy state"""
        self.position = 0
        self.last_signal = 0
        self.portfolio_value = 1.0
        self.trades = [] 