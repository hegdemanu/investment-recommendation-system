#!/usr/bin/env python3
"""
Stock Analysis Script using Archived Data
This script uses the archived MarketDataPipeline to analyze stock data
and generate forecasts using the Prophet model.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import argparse
import logging
from pathlib import Path
import glob

# Add the archive path to system path to resolve imports
archive_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          'archive/backend_backup_1743086055')
sys.path.append(archive_path)

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from data.market_data import MarketDataPipeline, calculate_technical_indicators
    logger.info("Successfully imported MarketDataPipeline from archived data")
except ImportError as e:
    logger.error(f"Failed to import from archived data: {e}")
    sys.exit(1)

# Import Prophet for forecasting
try:
    from prophet import Prophet
    logger.info("Successfully imported Prophet")
except ImportError:
    logger.error("Prophet not installed. Install using: pip install prophet")
    sys.exit(1)

def load_stock_data(stock_symbol):
    """
    Load stock data from the data_pipeline/stocks directory
    """
    # Look for files matching the stock symbol pattern
    stock_files = glob.glob(f"data_pipeline/stocks/*{stock_symbol}*.csv")
    
    if not stock_files:
        # Try looking in enhanced_data.csv
        logger.info(f"No specific file found for {stock_symbol}, trying enhanced_data.csv")
        enhanced_data = pd.read_csv("data_pipeline/enhanced_data.csv")
        
        # Check if the ticker exists in the data
        if 'ticker' in enhanced_data.columns and stock_symbol in enhanced_data['ticker'].unique():
            stock_data = enhanced_data[enhanced_data['ticker'] == stock_symbol].copy()
            logger.info(f"Found {len(stock_data)} records for {stock_symbol} in enhanced_data.csv")
            return stock_data
        else:
            logger.error(f"No data found for {stock_symbol}")
            return None
    
    # Load the first matching file
    logger.info(f"Loading data from {stock_files[0]}")
    stock_data = pd.read_csv(stock_files[0])
    
    # Convert date column to datetime if needed
    if 'Date' in stock_data.columns:
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
    return stock_data

def prepare_data_for_prophet(data):
    """
    Prepare data for Prophet forecasting
    """
    # Prophet requires columns named 'ds' and 'y'
    prophet_data = data.copy()
    
    # Check if we have the expected columns
    if 'Date' in prophet_data.columns and 'Close' in prophet_data.columns:
        prophet_data = prophet_data.rename(columns={'Date': 'ds', 'Close': 'y'})
    elif 'Date' in prophet_data.columns and 'Price' in prophet_data.columns:
        prophet_data = prophet_data.rename(columns={'Date': 'ds', 'Price': 'y'})
    else:
        # For enhanced_data.csv format
        prophet_data = prophet_data.rename(columns={'Date': 'ds', 'Price': 'y'})
    
    # Ensure ds is datetime
    prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
    
    return prophet_data

def run_prophet_forecast(data, ticker, periods=30):
    """
    Run Prophet forecast on the given data
    """
    # Prepare data for Prophet
    prophet_data = prepare_data_for_prophet(data)
    
    # Initialize and fit the model
    model = Prophet(yearly_seasonality=True, 
                  weekly_seasonality=True, 
                  daily_seasonality=False,
                  changepoint_prior_scale=0.05)
    
    model.fit(prophet_data)
    
    # Create future dataframe for prediction
    future = model.make_future_dataframe(periods=periods)
    
    # Make prediction
    forecast = model.predict(future)
    
    # Plot forecast
    fig = model.plot(forecast)
    plt.title(f"{ticker} Stock Price Forecast")
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{ticker}_forecast.png")
    
    # Plot components
    try:
        fig2 = model.plot_components(forecast)
        plt.savefig(output_dir / f"{ticker}_components.png")
    except Exception as e:
        logger.warning(f"Could not plot components: {e}")
    
    return forecast

def analyze_with_technical_indicators(data, ticker):
    """
    Add technical indicators and analyze the stock
    """
    # Ensure we have OHLC data
    required_cols = ['Open', 'High', 'Low', 'Close']
    
    # For enhanced_data.csv structure
    if 'Price' in data.columns and not all(col in data.columns for col in required_cols):
        # Create synthetic OHLC data if only Price is available
        data['Open'] = data['Price']
        data['High'] = data['Price'] * (1 + data['Price'].pct_change().abs().mean())
        data['Low'] = data['Price'] * (1 - data['Price'].pct_change().abs().mean())
        data['Close'] = data['Price']
    
    # Calculate technical indicators
    data_with_indicators = calculate_technical_indicators(data)
    
    # Plot key technical indicators
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    # Plot price and moving averages
    ax1.plot(data_with_indicators['Date'], data_with_indicators['Close'], label='Close Price')
    if 'MA20' in data_with_indicators.columns:
        ax1.plot(data_with_indicators['Date'], data_with_indicators['MA20'], label='20-day MA')
    ax1.set_title(f"{ticker} Price and Moving Average")
    ax1.legend()
    
    # Plot RSI
    if 'RSI' in data_with_indicators.columns:
        ax2.plot(data_with_indicators['Date'], data_with_indicators['RSI'], color='purple')
        ax2.axhline(y=70, color='r', linestyle='-', alpha=0.3)
        ax2.axhline(y=30, color='g', linestyle='-', alpha=0.3)
        ax2.set_title('RSI (14)')
        ax2.set_ylim(0, 100)
    
    # Plot MACD
    if 'MACD' in data_with_indicators.columns and 'Signal_Line' in data_with_indicators.columns:
        ax3.plot(data_with_indicators['Date'], data_with_indicators['MACD'], label='MACD')
        ax3.plot(data_with_indicators['Date'], data_with_indicators['Signal_Line'], label='Signal Line')
        ax3.set_title('MACD')
        ax3.legend()
    
    plt.tight_layout()
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{ticker}_technical_analysis.png")
    
    return data_with_indicators

def generate_trading_signals(data_with_indicators):
    """
    Generate trading signals based on technical indicators
    """
    signals = pd.DataFrame(index=data_with_indicators.index)
    signals['Price'] = data_with_indicators['Close']
    signals['Signal'] = 0.0
    
    # Generate signals based on RSI
    if 'RSI' in data_with_indicators.columns:
        signals.loc[data_with_indicators['RSI'] < 30, 'Signal'] = 1.0  # Buy signal
        signals.loc[data_with_indicators['RSI'] > 70, 'Signal'] = -1.0  # Sell signal
    
    # Generate signals based on MACD crossover
    if 'MACD' in data_with_indicators.columns and 'Signal_Line' in data_with_indicators.columns:
        signals.loc[data_with_indicators['MACD'] > data_with_indicators['Signal_Line'], 'MACD_Signal'] = 1.0  # Buy signal
        signals.loc[data_with_indicators['MACD'] < data_with_indicators['Signal_Line'], 'MACD_Signal'] = -1.0  # Sell signal
        
        # Combine signals
        if 'MACD_Signal' in signals.columns:
            signals['Signal'] = signals['Signal'] + signals['MACD_Signal']
    
    return signals

def main():
    parser = argparse.ArgumentParser(description='Stock Analysis Tool')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    args = parser.parse_args()
    
    # Load stock data
    stock_data = load_stock_data(args.ticker)
    
    if stock_data is None or len(stock_data) == 0:
        logger.error(f"No data found for ticker {args.ticker}")
        return
    
    logger.info(f"Loaded {len(stock_data)} data points for {args.ticker}")
    
    # Print preview of the data
    logger.info("Data preview:")
    logger.info(stock_data.head())
    
    # Generate technical analysis
    data_with_indicators = analyze_with_technical_indicators(stock_data, args.ticker)
    
    # Generate trading signals
    signals = generate_trading_signals(data_with_indicators)
    logger.info("Generated trading signals")
    
    # Run Prophet forecast
    forecast = run_prophet_forecast(stock_data, args.ticker, periods=args.days)
    
    # Print forecast summary
    last_price = stock_data.iloc[-1]['Close'] if 'Close' in stock_data.columns else stock_data.iloc[-1]['Price']
    forecast_price = forecast.iloc[-1]['yhat']
    expected_movement = (forecast_price / last_price - 1) * 100
    
    logger.info(f"\nForecast Summary for {args.ticker}:")
    logger.info(f"Last Price: ${last_price:.2f}")
    logger.info(f"Forecasted Price ({args.days} days): ${forecast_price:.2f}")
    logger.info(f"Expected Movement: {expected_movement:.2f}%")
    
    # Trading recommendation
    if expected_movement > 5:
        recommendation = "STRONG BUY"
    elif expected_movement > 2:
        recommendation = "BUY"
    elif expected_movement < -5:
        recommendation = "STRONG SELL"
    elif expected_movement < -2:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    
    logger.info(f"Trading Recommendation: {recommendation}")
    
    # Save recommendation to file
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / f"{args.ticker}_recommendation.txt", 'w') as f:
        f.write(f"Forecast Summary for {args.ticker}:\n")
        f.write(f"Last Price: ${last_price:.2f}\n")
        f.write(f"Forecasted Price ({args.days} days): ${forecast_price:.2f}\n")
        f.write(f"Expected Movement: {expected_movement:.2f}%\n")
        f.write(f"Trading Recommendation: {recommendation}\n")
    
    logger.info(f"Results saved to results/{args.ticker}_recommendation.txt")

if __name__ == "__main__":
    main() 