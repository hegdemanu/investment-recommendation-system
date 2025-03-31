#!/usr/bin/env python3
"""
Backtester for Trading Strategies
Evaluates strategy performance on historical data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtester:
    """Backtester for trading strategies"""
    
    def __init__(self, initial_capital: float = 10000.0):
        """Initialize backtester with initial capital"""
        self.initial_capital = initial_capital
        self.reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def backtest_strategy(self, strategy, data: pd.DataFrame) -> Dict:
        """Backtest a strategy on historical data"""
        logger.info(f"Backtesting strategy: {strategy.name}")
        
        # Reset strategy
        strategy.reset()
        
        # Run backtest
        result = strategy.backtest(data)
        
        # Enhance result with additional metrics
        self.calculate_additional_metrics(result)
        
        # Save backtest report
        self.save_report(strategy.name, result)
        
        return result
    
    def calculate_additional_metrics(self, result: Dict):
        """Calculate additional performance metrics"""
        returns = result.get("returns")
        if returns is None or len(returns) < 2:
            return
        
        # Calculate additional metrics
        
        # 1. Annualized return
        days = len(returns)
        if days > 0:
            total_return = result["total_return"]
            result["annualized_return"] = (1 + total_return) ** (252 / days) - 1
        
        # 2. Volatility
        daily_std = returns.std()
        result["volatility"] = daily_std * np.sqrt(252)
        
        # 3. Sortino ratio (downside risk)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0.0001
        if downside_std > 0:
            result["sortino_ratio"] = returns.mean() / downside_std * np.sqrt(252)
        else:
            result["sortino_ratio"] = 0
        
        # 4. Maximum consecutive losses
        streaks = []
        current_streak = 0
        for ret in returns:
            if ret < 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        if current_streak > 0:
            streaks.append(current_streak)
        result["max_consecutive_losses"] = max(streaks) if streaks else 0
        
        # 5. Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        result["profit_factor"] = gains / losses if losses > 0 else float('inf')
        
        # 6. Win rate
        wins = (returns > 0).sum()
        total_trades = len(result.get("trades", []))
        result["win_rate"] = wins / total_trades if total_trades > 0 else 0
        
        # 7. Average trade
        if total_trades > 0:
            result["avg_trade"] = returns.sum() / total_trades
        
        # 8. Return to drawdown ratio
        if result["max_drawdown"] != 0:
            result["return_to_drawdown"] = result["total_return"] / abs(result["max_drawdown"])
        else:
            result["return_to_drawdown"] = float('inf')
    
    def compare_strategies(self, strategies: List, data: pd.DataFrame) -> Dict:
        """Compare multiple strategies on the same dataset"""
        results = {}
        
        for strategy in strategies:
            results[strategy.name] = self.backtest_strategy(strategy, data)
        
        # Generate comparison report
        comparison = self.generate_comparison(results)
        
        # Save comparison report
        self.save_comparison(comparison)
        
        return comparison
    
    def generate_comparison(self, results: Dict) -> Dict:
        """Generate comparison report from backtest results"""
        metrics = [
            "total_return", "annualized_return", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "volatility", "win_rate", "profit_factor",
            "return_to_drawdown", "num_trades"
        ]
        
        comparison = {
            "strategies": list(results.keys()),
            "metrics": {},
            "timestamp": datetime.now().isoformat(),
            "best_strategy": None,
            "ranking": {}
        }
        
        # Extract metrics for comparison
        for metric in metrics:
            comparison["metrics"][metric] = {
                strategy_name: results[strategy_name].get(metric, 0)
                for strategy_name in results
            }
        
        # Calculate rankings for each metric
        for metric in metrics:
            values = [
                (strategy_name, results[strategy_name].get(metric, 0))
                for strategy_name in results
            ]
            
            # Sort based on whether higher is better
            reverse = metric not in ["max_drawdown", "volatility"]
            sorted_values = sorted(values, key=lambda x: x[1], reverse=reverse)
            
            comparison["ranking"][metric] = {
                strategy_name: rank + 1
                for rank, (strategy_name, _) in enumerate(sorted_values)
            }
        
        # Calculate overall ranking
        total_ranks = {
            strategy_name: sum(
                comparison["ranking"][metric][strategy_name] 
                for metric in metrics
            )
            for strategy_name in results
        }
        
        sorted_ranks = sorted(total_ranks.items(), key=lambda x: x[1])
        comparison["overall_ranking"] = {
            strategy_name: rank + 1
            for rank, (strategy_name, _) in enumerate(sorted_ranks)
        }
        
        # Determine best strategy
        if sorted_ranks:
            comparison["best_strategy"] = sorted_ranks[0][0]
        
        return comparison
    
    def save_report(self, strategy_name: str, result: Dict):
        """Save backtest report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{strategy_name}_{timestamp}.json"
        filepath = os.path.join(self.reports_dir, filename)
        
        # Prepare serializable result
        report = {
            "strategy_name": result["strategy_name"],
            "params": result["params"],
            "total_return": result["total_return"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"],
            "benchmark_return": result["benchmark_return"],
            "outperformance": result["outperformance"],
            "trades": result["trades"],
            "num_trades": result["num_trades"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Add additional metrics if available
        for metric in [
            "annualized_return", "volatility", "sortino_ratio",
            "max_consecutive_losses", "profit_factor", "win_rate",
            "avg_trade", "return_to_drawdown"
        ]:
            if metric in result:
                report[metric] = result[metric]
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved backtest report to {filepath}")
        except Exception as e:
            logger.error(f"Error saving backtest report: {e}")
    
    def save_comparison(self, comparison: Dict):
        """Save comparison report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{timestamp}.json"
        filepath = os.path.join(self.reports_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"Saved comparison report to {filepath}")
        except Exception as e:
            logger.error(f"Error saving comparison report: {e}")
    
    def generate_charts(self, result: Dict, output_dir: str = None):
        """Generate charts from backtest results"""
        if output_dir is None:
            output_dir = self.reports_dir
        
        strategy_name = result["strategy_name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Check if we have return data to plot
        if "cumulative_returns" not in result or "benchmark_cumulative" not in result:
            logger.warning("Missing return data for chart generation")
            return
        
        cumulative_returns = result["cumulative_returns"]
        benchmark_cumulative = result["benchmark_cumulative"]
        drawdown = result["drawdown"]
        
        # Create plot directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(cumulative_returns, label=strategy_name)
        plt.plot(benchmark_cumulative, label="Benchmark")
        plt.title(f"Cumulative Returns: {strategy_name}")
        plt.legend()
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        plt.plot(drawdown, label="Drawdown", color='red')
        plt.title(f"Drawdown: {strategy_name}")
        plt.legend()
        plt.grid(True)
        
        # Save chart
        chart_path = os.path.join(output_dir, f"{strategy_name}_{timestamp}.png")
        plt.savefig(chart_path)
        plt.close()
        logger.info(f"Saved performance chart to {chart_path}")
        
        # Generate trade chart
        self.generate_trade_chart(result, output_dir)
    
    def generate_trade_chart(self, result: Dict, output_dir: str):
        """Generate chart with trade entries and exits"""
        strategy_name = result["strategy_name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if "trades" not in result or "cumulative_returns" not in result:
            logger.warning("Missing trade data for chart generation")
            return
        
        trades = result["trades"]
        returns = result["cumulative_returns"]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.plot(returns, label="Equity Curve")
        
        # Plot trade entries and exits
        for trade in trades:
            date = pd.to_datetime(trade["date"])
            if date in returns.index:
                if trade["action"] == "BUY":
                    plt.scatter(date, returns.loc[date], s=100, c='g', marker='^')
                else:
                    plt.scatter(date, returns.loc[date], s=100, c='r', marker='v')
        
        plt.title(f"Trade Entries and Exits: {strategy_name}")
        plt.legend()
        plt.grid(True)
        
        # Save chart
        chart_path = os.path.join(output_dir, f"{strategy_name}_trades_{timestamp}.png")
        plt.savefig(chart_path)
        plt.close()
        logger.info(f"Saved trade chart to {chart_path}")
    
    def optimize_strategy(self, strategy_class, data: pd.DataFrame, param_grid: Dict) -> Dict:
        """Optimize strategy parameters using grid search"""
        logger.info(f"Optimizing strategy: {strategy_class.__name__}")
        
        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = list(product(*param_values))
        total_combinations = len(combinations)
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        results = []
        
        # Run backtest for each combination
        for i, combination in enumerate(combinations):
            params = {name: value for name, value in zip(param_names, combination)}
            
            # Create strategy instance with params
            strategy = strategy_class(**params)
            
            # Run backtest
            result = self.backtest_strategy(strategy, data)
            
            # Store result
            results.append({
                "params": params,
                "metrics": {
                    "total_return": result["total_return"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "max_drawdown": result["max_drawdown"],
                    "outperformance": result["outperformance"]
                }
            })
            
            logger.info(f"Combination {i+1}/{total_combinations} complete")
        
        # Find best parameters
        best_result = max(results, key=lambda x: x["metrics"]["sharpe_ratio"])
        
        # Save optimization results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_{strategy_class.__name__}_{timestamp}.json"
        filepath = os.path.join(self.reports_dir, filename)
        
        optimization_result = {
            "strategy": strategy_class.__name__,
            "param_grid": param_grid,
            "best_params": best_result["params"],
            "best_metrics": best_result["metrics"],
            "all_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(optimization_result, f, indent=2)
            logger.info(f"Saved optimization results to {filepath}")
        except Exception as e:
            logger.error(f"Error saving optimization results: {e}")
        
        return best_result 