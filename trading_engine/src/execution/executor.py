#!/usr/bin/env python3
"""
Trading Executor
Manages execution of trading strategies and simulation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import time
import threading
import json
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingExecutor:
    """Manages execution of trading strategies"""
    
    def __init__(self, strategies: List = None):
        """Initialize trading executor with strategies"""
        self.strategies = strategies or []
        self.positions = {}  # Current positions by symbol
        self.portfolio = {
            "cash": 10000.0,  # Initial cash
            "equity": 0.0,    # Value of positions
            "total": 10000.0  # Total portfolio value
        }
        self.trades = []
        self.performance_history = []
        self.is_running = False
        self.execution_thread = None
        self.last_update = datetime.now()
    
    def add_strategy(self, strategy):
        """Add a strategy to the executor"""
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_name: str):
        """Remove a strategy by name"""
        self.strategies = [s for s in self.strategies if s.name != strategy_name]
        logger.info(f"Removed strategy: {strategy_name}")
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for symbols"""
        prices = {}
        for symbol in symbols:
            # In a real system, you would use a market data provider
            # This is a simplified implementation that delegates to the strategy
            for strategy in self.strategies:
                if hasattr(strategy, 'symbol') and strategy.symbol == symbol and hasattr(strategy, 'get_current_price'):
                    price = strategy.get_current_price()
                    if price:
                        prices[symbol] = price
                        break
        return prices
    
    def update_portfolio(self):
        """Update portfolio value based on current positions and prices"""
        symbols = list(self.positions.keys())
        if not symbols:
            self.portfolio["equity"] = 0.0
            self.portfolio["total"] = self.portfolio["cash"]
            return
        
        prices = self.get_current_prices(symbols)
        equity_value = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                equity_value += position["quantity"] * prices[symbol]
        
        self.portfolio["equity"] = equity_value
        self.portfolio["total"] = self.portfolio["cash"] + equity_value
        
        # Record performance history
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "cash": self.portfolio["cash"],
            "equity": self.portfolio["equity"],
            "total": self.portfolio["total"]
        })
    
    def execute_trade(self, symbol: str, action: str, quantity: float, price: float, strategy_name: str):
        """Execute a trade and update portfolio"""
        if action not in ["BUY", "SELL"]:
            logger.error(f"Invalid action: {action}")
            return False
        
        trade_value = quantity * price
        commission = max(1.0, trade_value * 0.001)  # Simplified commission calculation
        
        if action == "BUY":
            # Check if we have enough cash
            if trade_value + commission > self.portfolio["cash"]:
                logger.warning(f"Insufficient cash for {action} {quantity} {symbol} at {price}")
                return False
            
            # Update cash
            self.portfolio["cash"] -= (trade_value + commission)
            
            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "quantity": quantity,
                    "avgPrice": price
                }
            else:
                total_quantity = self.positions[symbol]["quantity"] + quantity
                total_cost = (self.positions[symbol]["quantity"] * self.positions[symbol]["avgPrice"]) + trade_value
                self.positions[symbol] = {
                    "quantity": total_quantity,
                    "avgPrice": total_cost / total_quantity
                }
        
        elif action == "SELL":
            # Check if we have the position
            if symbol not in self.positions or self.positions[symbol]["quantity"] < quantity:
                logger.warning(f"Insufficient shares for {action} {quantity} {symbol}")
                return False
            
            # Update cash
            self.portfolio["cash"] += (trade_value - commission)
            
            # Update position
            remaining = self.positions[symbol]["quantity"] - quantity
            if remaining <= 0:
                del self.positions[symbol]
            else:
                self.positions[symbol]["quantity"] = remaining
        
        # Record the trade
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "value": trade_value,
            "commission": commission,
            "strategy": strategy_name
        }
        self.trades.append(trade)
        logger.info(f"Executed trade: {trade}")
        
        # Update portfolio
        self.update_portfolio()
        
        return True
    
    def process_signals(self):
        """Process signals from all strategies"""
        for strategy in self.strategies:
            # Check if the strategy has a symbol attribute (like MLStrategy)
            if not hasattr(strategy, 'symbol'):
                continue
                
            symbol = strategy.symbol
            current_signal = strategy.get_current_position()
            
            # Get prediction from strategy if available
            prediction = None
            if hasattr(strategy, 'get_next_prediction'):
                prediction = strategy.get_next_prediction()
                
            # Skip if no prediction or no signal
            if not prediction or prediction.get("signal") is None:
                continue
                
            signal = prediction["signal"]
            price = prediction.get("current_price")
            
            # Skip if no price available
            if not price:
                continue
                
            # Process signal if it's changed
            if signal != current_signal:
                # Close existing position if any
                if current_signal != 0:
                    # Determine quantity to sell
                    quantity = 0
                    if symbol in self.positions:
                        quantity = self.positions[symbol]["quantity"]
                        
                    if quantity > 0:
                        self.execute_trade(symbol, "SELL", quantity, price, strategy.name)
                        strategy.update_position(0)
                
                # Open new position if signal is non-zero
                if signal != 0:
                    # Calculate position size (10% of portfolio)
                    position_value = self.portfolio["total"] * 0.1
                    quantity = position_value / price
                    
                    if signal > 0:  # Buy signal
                        self.execute_trade(symbol, "BUY", quantity, price, strategy.name)
                        strategy.update_position(1)
                    elif signal < 0:  # Short signal (simplified)
                        # In a real system, short selling would be more complex
                        strategy.update_position(-1)
    
    def run_once(self):
        """Execute one cycle of the trading executor"""
        try:
            # Process signals from strategies
            self.process_signals()
            
            # Update portfolio value
            self.update_portfolio()
            
            # Save state
            self.save_state()
            
            self.last_update = datetime.now()
        except Exception as e:
            logger.error(f"Error in trading executor: {e}")
    
    def run_continuous(self, interval: int = 60):
        """Run trading executor continuously with specified interval in seconds"""
        self.is_running = True
        logger.info(f"Trading executor started with interval {interval} seconds")
        
        while self.is_running:
            self.run_once()
            time.sleep(interval)
    
    def start(self, interval: int = 60):
        """Start trading executor in a separate thread"""
        if self.is_running:
            logger.warning("Trading executor already running")
            return
            
        self.is_running = True
        self.execution_thread = threading.Thread(
            target=self.run_continuous,
            args=(interval,)
        )
        self.execution_thread.daemon = True
        self.execution_thread.start()
        logger.info("Trading executor started in background")
    
    def stop(self):
        """Stop trading executor"""
        self.is_running = False
        if self.execution_thread:
            self.execution_thread.join(timeout=5.0)
        logger.info("Trading executor stopped")
    
    def get_status(self) -> Dict:
        """Get current status of the trading executor"""
        return {
            "running": self.is_running,
            "lastUpdate": self.last_update.isoformat(),
            "portfolio": self.portfolio,
            "positions": self.positions,
            "strategies": [s.name for s in self.strategies],
            "tradeCount": len(self.trades)
        }
    
    def get_performance(self, days: int = 30) -> Dict:
        """Get performance over specified period"""
        if not self.performance_history:
            return {
                "returns": 0.0,
                "drawdown": 0.0,
                "sharpe": 0.0,
                "history": []
            }
        
        # Filter history for specified period
        cutoff = datetime.now() - timedelta(days=days)
        filtered_history = [
            entry for entry in self.performance_history
            if datetime.fromisoformat(entry["timestamp"]) >= cutoff
        ]
        
        if not filtered_history:
            return {
                "returns": 0.0,
                "drawdown": 0.0,
                "sharpe": 0.0,
                "history": []
            }
        
        # Calculate metrics
        start_value = filtered_history[0]["total"]
        end_value = filtered_history[-1]["total"]
        returns = (end_value - start_value) / start_value
        
        # Calculate drawdown
        peak = 0
        drawdown = 0
        for entry in filtered_history:
            if entry["total"] > peak:
                peak = entry["total"]
            current_drawdown = (peak - entry["total"]) / peak if peak > 0 else 0
            drawdown = max(drawdown, current_drawdown)
        
        # Calculate daily returns for Sharpe ratio
        daily_returns = []
        for i in range(1, len(filtered_history)):
            prev = filtered_history[i-1]["total"]
            curr = filtered_history[i]["total"]
            daily_returns.append((curr - prev) / prev if prev > 0 else 0)
        
        sharpe = 0
        if daily_returns:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        return {
            "returns": returns,
            "drawdown": drawdown,
            "sharpe": sharpe,
            "history": filtered_history
        }
    
    def save_state(self, filename: str = None):
        """Save executor state to file"""
        if not filename:
            filename = "trading_executor_state.json"
            
        state = {
            "portfolio": self.portfolio,
            "positions": self.positions,
            "trades": self.trades[-100:],  # Keep last 100 trades
            "performance": self.performance_history[-100:],  # Keep last 100 performance entries
            "lastUpdate": self.last_update.isoformat()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self, filename: str = None):
        """Load executor state from file"""
        if not filename:
            filename = "trading_executor_state.json"
            
        if not os.path.exists(filename):
            logger.warning(f"State file {filename} not found")
            return False
            
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
                
            self.portfolio = state.get("portfolio", self.portfolio)
            self.positions = state.get("positions", self.positions)
            self.trades = state.get("trades", self.trades)
            self.performance_history = state.get("performance", self.performance_history)
            
            if "lastUpdate" in state:
                self.last_update = datetime.fromisoformat(state["lastUpdate"])
                
            logger.info(f"Loaded state from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False 