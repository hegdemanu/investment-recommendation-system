import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
import yfinance as yf

# Add the project root to Python path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Prophet forecaster
from trading_engine.models.prophet_model_implementation import ProphetForecaster

def main():
    """
    Demo script to test the Prophet forecaster using real stock data
    """
    print("=" * 80)
    print("PROPHET FORECASTING DEMO")
    print("=" * 80)
    
    # Initialize the Prophet forecaster
    print("\nInitializing Prophet forecaster...")
    forecaster = ProphetForecaster()
    
    # Download historical data for a stock
    ticker = "AAPL"
    print(f"\nDownloading historical data for {ticker}...")
    
    # Use yfinance to download data
    try:
        # Get 2 years of historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2*365)
        
        # Download data
        stock_data = yf.download(
            ticker, 
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
        
        # Convert to a clean DataFrame format that Prophet can use
        data = stock_data.reset_index()
        
        # Rename columns to ensure they're clean
        data.columns = [str(col) if isinstance(col, tuple) else col for col in data.columns]
        
        print(f"Downloaded {len(data)} days of historical data.")
        
        # Display the first few rows of data
        print("\nPreview of the data:")
        print(data.head())
        
        # Check if data is valid
        if len(data) < 30:
            print("Error: Not enough data points for analysis.")
            return
            
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Generating sample data instead...")
        
        # Generate sample data if download fails
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        prices = np.linspace(100, 150, len(dates)) + np.random.normal(0, 5, len(dates))
        prices = prices + 10 * np.sin(np.linspace(0, 10, len(dates)))  # Add some seasonality
        
        data = pd.DataFrame({
            'Date': dates,
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        })
    
    # Split data into training and test sets
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"\nSplit data into {len(train_data)} training points and {len(test_data)} test points.")
    
    # Convert train_data and test_data to ensure they are regular DataFrames
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)
    
    # Train the Prophet model
    print("\nTraining Prophet model...")
    result = forecaster.train(train_data, target_col='Close')
    
    if result['success']:
        print("Prophet model trained successfully.")
        print(f"Model info: {result['model_info']}")
        
        # Generate future predictions
        forecast_days = 60
        print(f"\nGenerating {forecast_days}-day forecast...")
        forecast = forecaster.predict(periods=forecast_days)
        
        print("\nForecast preview:")
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        
        # Plot the forecast
        print("\nCreating forecast plot...")
        fig = forecaster.plot_forecast(forecast)
        
        # Save the plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/prophet_forecast.png')
        print("Forecast plot saved to results/prophet_forecast.png")
        
        # Plot the components
        print("\nCreating components plot...")
        fig_components = forecaster.plot_components(forecast)
        
        # Save the components plot
        plt.savefig('results/prophet_components.png')
        print("Components plot saved to results/prophet_components.png")
        
        # Evaluate the model on test data
        print("\nEvaluating model on test data...")
        eval_result = forecaster.evaluate(test_data)
        
        if eval_result['success']:
            metrics = eval_result['metrics']
            print("\nEvaluation Metrics:")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"MAPE: {metrics['mape']:.4f}%")
            
            # Plot actual vs predicted for test period
            plt.figure(figsize=(12, 6))
            
            # Get test period forecasts
            test_forecast = forecaster.predict(periods=0, include_history=True)
            
            # Convert test data dates to datetime for comparison
            test_dates = pd.to_datetime(test_data['Date'])
            min_test_date = test_dates.min()
            
            # Filter forecast for test period
            test_forecast = test_forecast[test_forecast['ds'] >= min_test_date]
            
            # Plot actual values
            plt.plot(test_data['Date'], test_data['Close'], 'b-', label='Actual')
            
            # Plot predicted values
            plt.plot(test_forecast['ds'], test_forecast['yhat'], 'r-', label='Predicted')
            plt.fill_between(test_forecast['ds'], 
                            test_forecast['yhat_lower'], 
                            test_forecast['yhat_upper'],
                            color='gray', alpha=0.2, label='Prediction Interval')
            
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Prophet Model Forecast vs Actual for {ticker}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('results/prophet_evaluation.png')
            print("Evaluation plot saved to results/prophet_evaluation.png")
            
        else:
            print(f"Error evaluating model: {eval_result.get('error', 'Unknown error')}")
        
        # Save the model
        print("\nSaving model...")
        forecaster.save_model(ticker=ticker)
        print(f"Model saved successfully")
        
    else:
        print(f"Error training model: {result.get('error', 'Unknown error')}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main() 