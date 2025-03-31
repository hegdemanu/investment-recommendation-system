import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Union, Callable
import logging
import os
from datetime import datetime, timedelta

from trading_engine.models.model_selector import ModelSelector, ModelType
from trading_engine.models.lstm_model import LSTMModel
from trading_engine.models.arima_model import ARIMAModel
from trading_engine.models.prophet_model import ProphetModel

class Backtester:
    """
    Backtesting framework for evaluating trading strategies and models
    """
    
    def __init__(self, 
                output_dir: str = "trading_engine/backtesting/results",
                initial_capital: float = 100000,
                transaction_cost: float = 0.001):
        """
        Initialize the backtester
        
        Args:
            output_dir: Directory to save backtest results
            initial_capital: Initial capital for backtesting
            transaction_cost: Transaction cost as a fraction of trade value
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    strategy: Callable, 
                    model_selector: ModelSelector = None,
                    start_date: str = None,
                    end_date: str = None,
                    name: str = None) -> Dict[str, Any]:
        """
        Run a backtest for a given strategy and model
        
        Args:
            data: Historical price data with Date and Close columns
            strategy: Strategy function that generates buy/sell signals
            model_selector: Model selector for predictions (optional)
            start_date: Start date for backtest
            end_date: End date for backtest
            name: Name for the backtest run
            
        Returns:
            Dictionary with backtest results
        """
        # Validate input data
        if 'Date' not in data.columns or 'Close' not in data.columns:
            raise ValueError("Data must contain 'Date' and 'Close' columns")
            
        # Filter data by date range if provided
        backtest_data = data.copy()
        if start_date:
            backtest_data = backtest_data[backtest_data['Date'] >= start_date]
        if end_date:
            backtest_data = backtest_data[backtest_data['Date'] <= end_date]
            
        # Sort data by date
        backtest_data = backtest_data.sort_values('Date').reset_index(drop=True)
        
        # Initialize results
        positions = np.zeros(len(backtest_data))
        capital = np.zeros(len(backtest_data))
        holdings = np.zeros(len(backtest_data))
        cash = np.zeros(len(backtest_data))
        
        # Set initial capital
        cash[0] = self.initial_capital
        capital[0] = self.initial_capital
        
        # Track trades
        trades = []
        
        # Run strategy day by day
        window_size = 60  # Use last 60 days for predictions
        
        for i in range(window_size, len(backtest_data)):
            # Get current window of data
            window = backtest_data.iloc[i-window_size:i]
            current_price = backtest_data.iloc[i]['Close']
            current_date = backtest_data.iloc[i]['Date']
            
            # Make prediction if model_selector is provided
            prediction = None
            if model_selector:
                try:
                    prediction = model_selector.predict(window)
                except Exception as e:
                    self.logger.warning(f"Error making prediction: {e}")
            
            # Generate signal using strategy
            signal = strategy(window, prediction)
            
            # Previous position and capital
            prev_position = positions[i-1] if i > 0 else 0
            prev_cash = cash[i-1] if i > 0 else self.initial_capital
            prev_holdings = holdings[i-1] if i > 0 else 0
            
            # Execute trades based on signal
            if signal == 1 and prev_position == 0:  # Buy
                # Calculate number of shares to buy with all available cash
                max_shares = prev_cash // (current_price * (1 + self.transaction_cost))
                positions[i] = max_shares
                cost = max_shares * current_price * (1 + self.transaction_cost)
                cash[i] = prev_cash - cost
                holdings[i] = max_shares * current_price
                
                # Record trade
                trades.append({
                    'date': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'shares': max_shares,
                    'cost': cost,
                    'prediction': prediction
                })
                
            elif signal == -1 and prev_position > 0:  # Sell
                # Sell all shares
                proceeds = prev_position * current_price * (1 - self.transaction_cost)
                positions[i] = 0
                cash[i] = prev_cash + proceeds
                holdings[i] = 0
                
                # Record trade
                trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': prev_position,
                    'proceeds': proceeds,
                    'prediction': prediction
                })
                
            else:  # Hold
                positions[i] = prev_position
                cash[i] = prev_cash
                holdings[i] = prev_position * current_price
            
            # Update capital
            capital[i] = cash[i] + holdings[i]
        
        # Calculate performance metrics
        returns = np.diff(capital) / capital[:-1]
        returns = np.insert(returns, 0, 0)
        
        # Calculate strategy performance metrics
        total_return = (capital[-1] - self.initial_capital) / self.initial_capital
        annualized_return = ((1 + total_return) ** (252 / len(backtest_data))) - 1
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = self._calculate_max_drawdown(capital)
        
        # Buy and hold benchmark
        buy_hold_capital = np.zeros(len(backtest_data))
        buy_hold_shares = self.initial_capital // backtest_data.iloc[window_size]['Close']
        
        for i in range(window_size, len(backtest_data)):
            buy_hold_capital[i] = buy_hold_shares * backtest_data.iloc[i]['Close']
        
        buy_hold_return = (buy_hold_capital[-1] - self.initial_capital) / self.initial_capital
        
        # Save results
        results = {
            'name': name or f"Backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_date': backtest_data['Date'].iloc[window_size],
            'end_date': backtest_data['Date'].iloc[-1],
            'initial_capital': self.initial_capital,
            'final_capital': capital[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'buy_hold_return': buy_hold_return,
            'total_trades': len(trades),
            'trades': trades,
            'capital_history': list(zip(backtest_data['Date'], capital)),
            'positions_history': list(zip(backtest_data['Date'], positions)),
        }
        
        # Plot results
        if name:
            self._plot_backtest_results(backtest_data, capital, buy_hold_capital, trades, name)
            
            # Save trade log
            self._save_trade_log(trades, name)
        
        return results
    
    def _calculate_max_drawdown(self, capital: np.ndarray) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            capital: Array of capital values
            
        Returns:
            Maximum drawdown as a positive fraction
        """
        # Find running maximum
        running_max = np.maximum.accumulate(capital)
        # Calculate drawdown
        drawdown = (running_max - capital) / running_max
        # Return maximum drawdown
        return np.max(drawdown)
    
    def _plot_backtest_results(self, 
                              data: pd.DataFrame, 
                              capital: np.ndarray, 
                              buy_hold_capital: np.ndarray, 
                              trades: List[Dict], 
                              name: str) -> None:
        """
        Plot backtest results
        
        Args:
            data: Historical price data
            capital: Array of capital values
            buy_hold_capital: Array of buy and hold capital values
            trades: List of trades
            name: Name for the plot
        """
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot price
        ax1.plot(data['Date'], data['Close'], label='Price', color='black', alpha=0.5)
        
        # Plot buy/sell points
        buy_dates = [trade['date'] for trade in trades if trade['action'] == 'BUY']
        buy_prices = [trade['price'] for trade in trades if trade['action'] == 'BUY']
        sell_dates = [trade['date'] for trade in trades if trade['action'] == 'SELL']
        sell_prices = [trade['price'] for trade in trades if trade['action'] == 'SELL']
        
        ax1.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='Buy')
        ax1.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='Sell')
        
        ax1.set_title(f'Backtest Results - {name}')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot portfolio value vs buy and hold
        window_size = 60
        ax2.plot(data['Date'][window_size:], capital[window_size:], label='Strategy', color='blue')
        ax2.plot(data['Date'][window_size:], buy_hold_capital[window_size:], label='Buy & Hold', color='orange', linestyle='--')
        
        ax2.set_title('Portfolio Performance')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{name}.png"))
        plt.close()
    
    def _save_trade_log(self, trades: List[Dict], name: str) -> None:
        """
        Save trade log to CSV
        
        Args:
            trades: List of trades
            name: Name for the file
        """
        if not trades:
            return
        
        # Convert to DataFrame
        trade_df = pd.DataFrame(trades)
        
        # Save to CSV
        trade_df.to_csv(os.path.join(self.output_dir, f"{name}_trades.csv"), index=False)
    
    def compare_models(self, 
                     data: pd.DataFrame, 
                     strategy: Callable, 
                     start_date: str = None,
                     end_date: str = None) -> Dict[str, Any]:
        """
        Compare performance of different models
        
        Args:
            data: Historical price data
            strategy: Strategy function
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Dictionary with comparison results
        """
        # Create models
        lstm_model = LSTMModel()
        arima_model = ARIMAModel()
        prophet_model = ProphetModel()
        
        # Get ticker from data if available
        ticker = data['ticker'].iloc[0] if 'ticker' in data.columns else 'UNKNOWN'
        
        # Train models on data before start_date if provided
        train_data = data.copy()
        if start_date:
            train_data = train_data[train_data['Date'] < start_date]
        
        # Train models
        try:
            lstm_model.train(train_data, ticker)
        except Exception as e:
            self.logger.warning(f"Error training LSTM model: {e}")
            
        try:
            arima_model.train(train_data, ticker)
        except Exception as e:
            self.logger.warning(f"Error training ARIMA model: {e}")
            
        try:
            prophet_model.train(train_data, ticker)
        except Exception as e:
            self.logger.warning(f"Error training Prophet model: {e}")
        
        # Create model selectors
        lstm_selector = ModelSelector()
        lstm_selector.add_model(ModelType.LSTM, lstm_model, "LSTM")
        
        arima_selector = ModelSelector()
        arima_selector.add_model(ModelType.ARIMA, arima_model, "ARIMA")
        
        prophet_selector = ModelSelector()
        prophet_selector.add_model(ModelType.PROPHET, prophet_model, "PROPHET")
        
        # Create ensemble selector
        ensemble_selector = ModelSelector()
        ensemble_selector.add_model(ModelType.LSTM, lstm_model, "LSTM")
        ensemble_selector.add_model(ModelType.ARIMA, arima_model, "ARIMA")
        ensemble_selector.add_model(ModelType.PROPHET, prophet_model, "PROPHET")
        
        # Run backtests
        lstm_results = self.run_backtest(
            data, strategy, lstm_selector, start_date, end_date, f"{ticker}_LSTM"
        )
        
        arima_results = self.run_backtest(
            data, strategy, arima_selector, start_date, end_date, f"{ticker}_ARIMA"
        )
        
        prophet_results = self.run_backtest(
            data, strategy, prophet_selector, start_date, end_date, f"{ticker}_PROPHET"
        )
        
        ensemble_results = self.run_backtest(
            data, strategy, ensemble_selector, start_date, end_date, f"{ticker}_ENSEMBLE"
        )
        
        # Create comparison chart
        self._plot_model_comparison([
            lstm_results,
            arima_results,
            prophet_results,
            ensemble_results
        ], ticker)
        
        return {
            'lstm': lstm_results,
            'arima': arima_results,
            'prophet': prophet_results,
            'ensemble': ensemble_results
        }
        
    def _plot_model_comparison(self, results: List[Dict], ticker: str) -> None:
        """
        Plot comparison of model performances
        
        Args:
            results: List of backtest results
            ticker: Ticker symbol
        """
        # Extract data for plotting
        names = [r['name'].split('_')[-1] for r in results]
        returns = [r['total_return'] * 100 for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        max_drawdowns = [r['max_drawdown'] * 100 for r in results]
        
        # Create figure with 1 row and 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot returns
        axes[0].bar(names, returns)
        axes[0].set_title('Total Return (%)')
        axes[0].set_ylabel('Return (%)')
        axes[0].grid(True, axis='y')
        
        # Plot Sharpe ratios
        axes[1].bar(names, sharpe_ratios)
        axes[1].set_title('Sharpe Ratio')
        axes[1].grid(True, axis='y')
        
        # Plot max drawdowns
        axes[2].bar(names, max_drawdowns)
        axes[2].set_title('Maximum Drawdown (%)')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].grid(True, axis='y')
        
        # Set overall title
        plt.suptitle(f'Model Comparison for {ticker}', fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(os.path.join(self.output_dir, f"{ticker}_model_comparison.png"))
        plt.close()

# Example strategy functions
def simple_moving_average_strategy(data: pd.DataFrame, prediction: float = None) -> int:
    """
    Simple moving average crossover strategy
    
    Args:
        data: Historical price data with Close column
        prediction: Model prediction (optional)
        
    Returns:
        1 for buy, -1 for sell, 0 for hold
    """
    if len(data) < 50:
        return 0
    
    # Calculate short and long moving averages
    short_ma = data['Close'][-20:].mean()
    long_ma = data['Close'][-50:].mean()
    
    # Current position
    current_price = data['Close'].iloc[-1]
    
    # Generate signals
    if short_ma > long_ma:
        return 1  # Buy
    elif short_ma < long_ma:
        return -1  # Sell
    else:
        return 0  # Hold

def prediction_based_strategy(data: pd.DataFrame, prediction: float = None) -> int:
    """
    Strategy based on model predictions
    
    Args:
        data: Historical price data with Close column
        prediction: Model prediction
        
    Returns:
        1 for buy, -1 for sell, 0 for hold
    """
    if prediction is None:
        return 0
    
    current_price = data['Close'].iloc[-1]
    
    # Buy if predicted price is higher by at least 1%
    if prediction > current_price * 1.01:
        return 1
    # Sell if predicted price is lower by at least 1%
    elif prediction < current_price * 0.99:
        return -1
    else:
        return 0 