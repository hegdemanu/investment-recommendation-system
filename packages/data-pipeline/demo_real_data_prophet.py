import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import argparse

# Add the project root to Python path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Prophet forecaster
from trading_engine.models.prophet_model_implementation import ProphetForecaster

def fix_yfinance_data(data):
    """
    Fix the data format from yfinance to work with Prophet
    
    Args:
        data: DataFrame from yfinance
        
    Returns:
        Fixed DataFrame
    """
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Fix any tuple column names from yfinance
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    # Ensure column names are strings
    data.columns = [str(col) for col in data.columns]
    
    return data

def main():
    """
    Script to perform stock market forecasting using real data and Prophet
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Forecast stock prices using Prophet and real data')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol to analyze (e.g., AAPL)')
    parser.add_argument('--days', type=int, default=60, help='Number of days to forecast')
    parser.add_argument('--years', type=int, default=2, help='Years of historical data to use')
    args = parser.parse_args()
    
    ticker = args.ticker
    forecast_days = args.days
    years_data = args.years
    
    print("=" * 80)
    print(f"REAL DATA STOCK FORECASTING FOR {ticker}")
    print("=" * 80)
    
    # Initialize the Prophet forecaster
    print("\nInitializing Prophet forecaster...")
    forecaster = ProphetForecaster()
    
    # Download historical data for the stock
    print(f"\nDownloading {years_data} years of historical data for {ticker}...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_data)
    
    # Download data from yfinance
    try:
        stock_data = yf.download(
            ticker, 
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        # Check if data is valid
        if len(stock_data) < 30:
            print(f"Error: Not enough data points for {ticker}. Received only {len(stock_data)} days of data.")
            return
            
        print(f"Successfully downloaded {len(stock_data)} days of historical data.")
        
        # Fix the data format
        data = fix_yfinance_data(stock_data)
        
        # Display sample of the data
        print("\nPreview of the data:")
        print(data.head())
        
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return
    
    # Split data into training and testing sets (80/20 split)
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"\nSplit data into {len(train_data)} training points and {len(test_data)} test points.")
    
    # Train the Prophet model
    print(f"\nTraining Prophet model on {ticker} data...")
    
    # Fine-tune parameters for financial data
    model_params = {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "changepoint_range": 0.9,
        "daily_seasonality": False,
        "weekly_seasonality": True,
        "yearly_seasonality": True
    }
    
    train_result = forecaster.train(train_data, target_col='Close', params=model_params)
    
    if not train_result['success']:
        print(f"Error training model: {train_result.get('error', 'Unknown error')}")
        return
        
    print("Prophet model trained successfully.")
    
    # Save trained model
    model_dir = os.path.join('trading_engine/models/prophet')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{ticker}_prophet_model.json")
    forecaster.save_model(path=model_path)
    print(f"Model saved to {model_path}")
    
    # Generate forecast
    print(f"\nGenerating {forecast_days}-day forecast for {ticker}...")
    forecast = forecaster.predict(periods=forecast_days)
    
    print("\nForecast preview (last 5 days):")
    forecast_tail = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    forecast_tail['ds'] = forecast_tail['ds'].dt.strftime('%Y-%m-%d')
    print(forecast_tail)
    
    # Calculate expected price movement
    last_close = data['Close'].iloc[-1]
    forecasted_price = forecast['yhat'].iloc[-1]
    price_change = ((forecasted_price - last_close) / last_close) * 100
    
    print(f"\nLast Close Price: ${last_close:.2f}")
    print(f"Forecasted Price ({forecast_days} days): ${forecasted_price:.2f}")
    print(f"Expected Movement: {price_change:.2f}%")
    
    # Evaluate on test data
    print("\nEvaluating model on test data...")
    eval_result = forecaster.evaluate(test_data)
    
    if not eval_result['success']:
        print(f"Error evaluating model: {eval_result.get('error', 'Unknown error')}")
    else:
        metrics = eval_result['metrics']
        print("\nPerformance Metrics:")
        print(f"RMSE: ${metrics['rmse']:.2f}")
        print(f"MAE: ${metrics['mae']:.2f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Create output directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot the forecast
    plt.figure(figsize=(14, 7))
    
    # Plot historical data
    plt.plot(data['Date'], data['Close'], 'b-', label='Historical Close Price')
    
    # Plot test data
    plt.plot(test_data['Date'], test_data['Close'], 'g-', label='Test Data')
    
    # Get future dates beyond the historical data
    last_date = data['Date'].max()
    future_dates = pd.date_range(start=last_date, periods=forecast_days+1)[1:]
    future_forecast = forecast[forecast['ds'] > pd.to_datetime(last_date)]
    
    # Plot the forecast
    plt.plot(future_forecast['ds'], future_forecast['yhat'], 'r-', label='Forecast')
    plt.fill_between(
        future_forecast['ds'], 
        future_forecast['yhat_lower'], 
        future_forecast['yhat_upper'],
        color='red', alpha=0.2, label='Prediction Interval'
    )
    
    # Format the plot
    plt.title(f'{ticker} Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    forecast_plot_path = os.path.join(results_dir, f'{ticker}_forecast.png')
    plt.savefig(forecast_plot_path)
    print(f"Forecast visualization saved to {forecast_plot_path}")
    
    # Plot forecast components
    components_fig = forecaster.plot_components(forecast)
    components_plot_path = os.path.join(results_dir, f'{ticker}_components.png')
    components_fig.savefig(components_plot_path)
    print(f"Forecast components saved to {components_plot_path}")
    
    print("\nForecasting completed successfully!")

if __name__ == "__main__":
    main() 