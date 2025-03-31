# Trading Engine for Investment Recommendation System

This module provides a complete trading engine that integrates with ML models to generate trading signals, execute trades, and backtest strategies.

## Features

- ML-based trading strategies that leverage predictions from the ML service
- Trading strategy framework with a common interface
- Trading executor for simulating and executing trades
- Backtesting framework with comprehensive performance metrics
- Strategy optimization through parameter grid search
- Performance reporting and visualization

## Architecture

The trading engine consists of several key components:

### Strategies

- `BaseStrategy`: The abstract base class that defines the interface for all trading strategies
- `MLStrategy`: A strategy that uses ML model predictions to generate trading signals

### Execution

- `TradingExecutor`: Manages the execution of trading strategies and maintains portfolio state

### Backtesting

- `Backtester`: Evaluates strategy performance on historical data and generates reports

## Usage

### Basic Backtesting

To backtest a strategy on historical data:

```python
from src.strategies.ml_strategy import MLStrategy
from src.backtest.backtester import Backtester
import yfinance as yf

# Get historical data
data = yf.download("AAPL", period="2y")

# Create strategy
strategy = MLStrategy(
    symbol="AAPL",
    model_type="ENSEMBLE",
    confidence_threshold=0.02,
    stop_loss=0.05,
    take_profit=0.10
)

# Run backtest
backtester = Backtester()
result = backtester.backtest_strategy(strategy, data)
backtester.generate_charts(result)
```

### Strategy Optimization

To find optimal parameters for a strategy:

```python
# Define parameter grid
param_grid = {
    "symbol": ["AAPL"],
    "model_type": ["ENSEMBLE"],
    "confidence_threshold": [0.01, 0.02, 0.03],
    "stop_loss": [0.03, 0.05, 0.07],
    "take_profit": [0.06, 0.10, 0.15]
}

# Run optimization
best_result = backtester.optimize_strategy(MLStrategy, data, param_grid)
```

### Strategy Comparison

To compare multiple strategies:

```python
# Create strategies
strategies = [
    MLStrategy(symbol="AAPL", model_type="LSTM"),
    MLStrategy(symbol="AAPL", model_type="PROPHET"),
    MLStrategy(symbol="AAPL", model_type="ENSEMBLE")
]

# Run comparison
comparison = backtester.compare_strategies(strategies, data)
```

### Trading Simulation

To simulate trading with a strategy:

```python
from src.execution.executor import TradingExecutor

# Create strategy
strategy = MLStrategy(symbol="AAPL", model_type="ENSEMBLE")

# Create executor
executor = TradingExecutor()
executor.add_strategy(strategy)

# Run simulation
for i in range(len(data) - 1):
    current_data = data.iloc[:i+1]
    signal = strategy.generate_signal(current_data)
    strategy.update_position(signal)
    executor.process_signals()
    executor.update_portfolio()

# Get results
status = executor.get_status()
performance = executor.get_performance()
```

## Command Line Interface

The trading engine includes a CLI for running common tasks:

```bash
# Run a backtest
python main.py backtest --symbol AAPL --model ENSEMBLE --period 2y

# Optimize a strategy
python main.py optimize --symbol AAPL --model LSTM --period 2y

# Compare strategies
python main.py compare --symbol AAPL --period 2y

# Run a simulation
python main.py simulate --symbol AAPL --model ENSEMBLE --days 30
```

## Creating Custom Strategies

You can create custom strategies by inheriting from `BaseStrategy`:

```python
from src.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, name, params=None):
        super().__init__(name, params)
        # Initialize your strategy-specific attributes
    
    def generate_signal(self, data):
        # Implement your signal generation logic
        # Return 1 for buy, 0 for neutral, -1 for sell
        return signal
```

## Performance Metrics

The backtester calculates a comprehensive set of performance metrics:

- **Total Return**: Overall return of the strategy
- **Annualized Return**: Return annualized to a yearly rate
- **Sharpe Ratio**: Return relative to risk (volatility)
- **Sortino Ratio**: Return relative to downside risk
- **Max Drawdown**: Maximum peak-to-trough decline
- **Volatility**: Annualized standard deviation of returns
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Return to Drawdown Ratio**: Return relative to maximum drawdown
- **Max Consecutive Losses**: Longest streak of losing trades

## Reports and Charts

Backtest results are saved to the `reports` directory, including:

- JSON reports with detailed performance metrics
- Performance charts showing equity curve and drawdown
- Trade charts showing entry and exit points
- Optimization results with best parameters
- Strategy comparison reports

## Integration with ML Service

The trading engine integrates with the ML service to get predictions:

```python
# In MLStrategy.get_predictions
response = requests.post(
    f"{self.ml_service_url}/predict",
    json={
        "symbol": self.symbol,
        "modelType": self.model_type,
        "horizon": self.holding_period
    }
)
```

Ensure the ML service is running on the expected URL (default: http://localhost:5001). 