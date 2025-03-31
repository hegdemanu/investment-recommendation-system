#!/usr/bin/env python3
"""
Trading Engine Demo Script
Demonstrates the trading engine with ML-based strategies
"""

import pandas as pd
import numpy as np
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
import yfinance as yf

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import trading engine components
from src.strategies.ml_strategy import MLStrategy
from src.execution.executor import TradingExecutor
from src.backtest.backtester import Backtester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_stock_data(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Get historical stock data"""
    try:
        data = yf.download(symbol, period=period, interval=interval, progress=False)
        if data.empty:
            logger.error(f"No data returned for {symbol}")
            return pd.DataFrame()
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def run_backtest(symbol: str, model_type: str = "LSTM", period: str = "2y"):
    """Run backtest for a strategy"""
    logger.info(f"Running backtest for {symbol} using {model_type} model")
    
    # Get historical data
    data = get_stock_data(symbol, period=period)
    if data.empty:
        logger.error("Cannot run backtest with empty data")
        return
    
    # Create strategy
    strategy = MLStrategy(
        symbol=symbol,
        model_type=model_type,
        confidence_threshold=0.02,
        stop_loss=0.05,
        take_profit=0.10,
        holding_period=5
    )
    
    # Create backtester
    backtester = Backtester()
    
    # Run backtest
    result = backtester.backtest_strategy(strategy, data)
    
    # Generate charts
    backtester.generate_charts(result)
    
    # Print results
    print("\n----- Backtest Results -----")
    print(f"Strategy: {result['strategy_name']}")
    print(f"Total Return: {result['total_return']:.2%}")
    print(f"Annualized Return: {result.get('annualized_return', 0):.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"Win Rate: {result.get('win_rate', 0):.2%}")
    print(f"Profit Factor: {result.get('profit_factor', 0):.2f}")
    print(f"Benchmark Return: {result['benchmark_return']:.2%}")
    print(f"Outperformance: {result['outperformance']:.2%}")
    print(f"Number of Trades: {result['num_trades']}")
    print("----------------------------\n")

def optimize_strategy(symbol: str, model_type: str = "LSTM", period: str = "2y"):
    """Optimize strategy parameters"""
    logger.info(f"Optimizing strategy for {symbol}")
    
    # Get historical data
    data = get_stock_data(symbol, period=period)
    if data.empty:
        logger.error("Cannot optimize with empty data")
        return
    
    # Define parameter grid
    param_grid = {
        "symbol": [symbol],
        "model_type": [model_type],
        "confidence_threshold": [0.01, 0.02, 0.03, 0.05],
        "stop_loss": [0.03, 0.05, 0.07],
        "take_profit": [0.06, 0.10, 0.15],
        "holding_period": [3, 5, 7, 10]
    }
    
    # Create backtester
    backtester = Backtester()
    
    # Run optimization
    best_result = backtester.optimize_strategy(MLStrategy, data, param_grid)
    
    # Print results
    print("\n----- Optimization Results -----")
    print(f"Best Parameters:")
    for param, value in best_result["params"].items():
        print(f"  {param}: {value}")
    
    print("\nPerformance Metrics:")
    for metric, value in best_result["metrics"].items():
        if metric in ["total_return", "max_drawdown", "outperformance"]:
            print(f"  {metric}: {value:.2%}")
        else:
            print(f"  {metric}: {value:.2f}")
    print("-------------------------------\n")

def compare_strategies(symbol: str, period: str = "2y"):
    """Compare different strategy configurations"""
    logger.info(f"Comparing strategies for {symbol}")
    
    # Get historical data
    data = get_stock_data(symbol, period=period)
    if data.empty:
        logger.error("Cannot compare strategies with empty data")
        return
    
    # Create strategies
    strategies = [
        MLStrategy(
            symbol=symbol,
            model_type="LSTM",
            confidence_threshold=0.02,
            stop_loss=0.05,
            take_profit=0.10,
            holding_period=5
        ),
        MLStrategy(
            symbol=symbol,
            model_type="PROPHET",
            confidence_threshold=0.03,
            stop_loss=0.04,
            take_profit=0.12,
            holding_period=7
        ),
        MLStrategy(
            symbol=symbol,
            model_type="ENSEMBLE",
            confidence_threshold=0.02,
            stop_loss=0.05,
            take_profit=0.10,
            holding_period=5
        )
    ]
    
    # Create backtester
    backtester = Backtester()
    
    # Run comparison
    comparison = backtester.compare_strategies(strategies, data)
    
    # Print results
    print("\n----- Strategy Comparison -----")
    print(f"Best Strategy: {comparison['best_strategy']}")
    print("\nMetrics by Strategy:")
    
    for metric in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"]:
        print(f"\n{metric}:")
        for strategy, value in comparison["metrics"][metric].items():
            if metric in ["total_return", "max_drawdown"]:
                print(f"  {strategy}: {value:.2%}")
            else:
                print(f"  {strategy}: {value:.2f}")
    
    print("\nOverall Ranking:")
    for strategy, rank in sorted(comparison["overall_ranking"].items(), key=lambda x: x[1]):
        print(f"  {rank}. {strategy}")
    print("------------------------------\n")

def run_simulation(symbol: str, model_type: str = "LSTM", days: int = 30):
    """Run trading simulation"""
    logger.info(f"Running trading simulation for {symbol}")
    
    # Create strategy
    strategy = MLStrategy(
        symbol=symbol,
        model_type=model_type,
        confidence_threshold=0.02,
        stop_loss=0.05,
        take_profit=0.10,
        holding_period=5
    )
    
    # Create executor
    executor = TradingExecutor()
    
    # Add strategy
    executor.add_strategy(strategy)
    
    # Load historical data for simulation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if data.empty:
        logger.error("Cannot run simulation with empty data")
        return
    
    # Run simulation
    for i in range(len(data) - 1):
        # Use data up to this point
        current_data = data.iloc[:i+1]
        
        # Update strategy position based on data
        signal = strategy.generate_signal(current_data)
        strategy.update_position(signal)
        
        # Process signals
        executor.process_signals()
        
        # Update portfolio
        executor.update_portfolio()
    
    # Get final results
    status = executor.get_status()
    performance = executor.get_performance(days=days)
    
    # Print results
    print("\n----- Simulation Results -----")
    print(f"Initial Capital: ${10000:.2f}")
    print(f"Final Portfolio Value: ${status['portfolio']['total']:.2f}")
    print(f"Return: {(status['portfolio']['total'] / 10000 - 1):.2%}")
    print(f"Current Positions:")
    
    for symbol, position in status['positions'].items():
        print(f"  {symbol}: {position['quantity']:.2f} shares at avg. price ${position['avgPrice']:.2f}")
    
    print(f"\nTrades: {status['tradeCount']}")
    print("----------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Trading Engine Demo")
    parser.add_argument("action", choices=["backtest", "optimize", "compare", "simulate"], 
                        help="Action to perform")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol to trade")
    parser.add_argument("--model", default="ENSEMBLE", choices=["LSTM", "ARIMA", "PROPHET", "ENSEMBLE"],
                        help="ML model type")
    parser.add_argument("--period", default="2y", help="Data period for backtest")
    parser.add_argument("--days", type=int, default=30, help="Number of days for simulation")
    
    args = parser.parse_args()
    
    if args.action == "backtest":
        run_backtest(args.symbol, args.model, args.period)
    elif args.action == "optimize":
        optimize_strategy(args.symbol, args.model, args.period)
    elif args.action == "compare":
        compare_strategies(args.symbol, args.period)
    elif args.action == "simulate":
        run_simulation(args.symbol, args.model, args.days)

if __name__ == "__main__":
    main() 