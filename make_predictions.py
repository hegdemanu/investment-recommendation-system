"""
Make predictions using the trained LSTM models.
This script:
1. Loads trained models for each stock
2. Makes predictions for the next 30 days
3. Visualizes and saves the predictions
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from src.model_trainer import ModelTrainer

# Configure directories
MODELS_DIR = "./models"
RESULTS_DIR = "./results/predictions"
DATA_DIR = "./data/stocks"

# Create output directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(filename):
    """Load and preprocess data from a CSV file."""
    print(f"Loading data from {filename}")
    
    try:
        # Load data
        data = pd.read_csv(filename)
        
        # Convert date to datetime
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Extract ticker from filename if not in data
        if 'ticker' not in data.columns:
            ticker = os.path.basename(filename).split('_')[0]
            data['ticker'] = ticker
            
        # Process numeric columns with commas
        for col in ['Price', 'Open', 'High', 'Low']:
            if col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = data[col].str.replace(',', '').astype(float)
                # Ensure column is float type
                data[col] = data[col].astype(float)
                
        # Process Vol. column with K, M suffixes
        if 'Vol.' in data.columns:
            if data['Vol.'].dtype == 'object':
                # Replace missing values with NaN
                data['Vol.'] = data['Vol.'].replace('-', np.nan).replace('', np.nan)
                
                # Process K and M suffixes and convert to numeric
                data.loc[data['Vol.'].notna(), 'Vol.'] = data.loc[data['Vol.'].notna(), 'Vol.'].apply(
                    lambda x: float(str(x).replace('K', '')) * 1000 if isinstance(x, str) and 'K' in x 
                    else float(str(x).replace('M', '')) * 1000000 if isinstance(x, str) and 'M' in x
                    else float(x) if x and pd.notna(x) else np.nan
                )
            
            # Ensure column is float type
            data['Vol.'] = pd.to_numeric(data['Vol.'], errors='coerce')
            
        # Process Change % column
        if 'Change %' in data.columns:
            if data['Change %'].dtype == 'object':
                data['Change %'] = data['Change %'].str.replace('%', '').astype(float) / 100
            # Ensure column is float type
            data['Change %'] = data['Change %'].astype(float)
            
        # Check for NaN values and handle them
        for col in data.columns:
            if data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if data[col].isna().any():
                    print(f"Found {data[col].isna().sum()} NaN values in column {col}, filling with mean")
                    data[col] = data[col].fillna(data[col].mean())
            
        print(f"Loaded {len(data)} rows with columns: {data.columns.tolist()}")
        return data
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def predict_future(model_trainer, data, ticker):
    """Make future predictions for a specific stock."""
    try:
        # Load model components
        model, scaler, features = model_trainer.load_model(ticker)
        
        if model is None:
            print(f"Failed to load model for {ticker}")
            return None
        
        # Filter data for this ticker
        ticker_data = data[data['ticker'] == ticker].copy()
        
        # Prepare sequences for prediction
        last_sequence, used_features = model_trainer.prepare_sequence_for_prediction(
            ticker_data, 
            features, 
            scaler
        )
        
        if last_sequence is None:
            print(f"Failed to prepare sequence for {ticker}")
            return None
            
        # Generate predictions for the next 30 days
        days_ahead = 30
        predictions = model_trainer.predict_future(model, last_sequence, scaler, used_features, days=days_ahead)
        
        if len(predictions) == 0:
            print(f"Failed to generate predictions for {ticker}")
            return None
            
        # Get prediction horizons info for best horizon
        horizons_path = os.path.join(MODELS_DIR, f"{ticker}_horizons.pkl")
        best_horizon = 1  # Default if not found
        
        if os.path.exists(horizons_path):
            import pickle
            with open(horizons_path, 'rb') as f:
                horizons_info = pickle.load(f)
            best_horizon = horizons_info['best_horizon']
        
        # Create future dates
        last_date = ticker_data['Date'].max()
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        # Create DataFrame with predictions
        prediction_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': predictions,
            'ticker': ticker
        })
        
        # Plot historical data and predictions
        plt.figure(figsize=(12, 6))
        
        # Plot historical prices
        plt.plot(ticker_data['Date'], ticker_data['Price'], label='Historical', color='blue')
        
        # Plot predictions
        plt.plot(prediction_df['Date'], prediction_df['Predicted_Price'], label=f'Predictions (Best horizon: {best_horizon} days)', 
                color='red', linestyle='--')
        
        # Add vertical line at the prediction start
        plt.axvline(x=last_date, color='green', linestyle='-', alpha=0.5)
        
        # Format plot
        plt.title(f'LSTM Price Predictions for {ticker} - Next {days_ahead} Days')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Adjust x-axis to show more dates clearly
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(RESULTS_DIR, f"{ticker}_future_prediction.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Save prediction data to CSV
        csv_path = os.path.join(RESULTS_DIR, f"{ticker}_predictions.csv")
        prediction_df.to_csv(csv_path, index=False)
        
        # Calculate expected return
        start_price = ticker_data['Price'].iloc[-1]
        end_price = prediction_df['Predicted_Price'].iloc[-1]
        expected_return = (end_price / start_price - 1) * 100
        
        result = {
            'ticker': ticker,
            'best_horizon_days': best_horizon,
            'days_predicted': days_ahead,
            'start_date': last_date.strftime('%Y-%m-%d'),
            'end_date': future_dates[-1].strftime('%Y-%m-%d'),
            'start_price': float(start_price),
            'end_price': float(end_price),
            'expected_return_percent': float(expected_return),
            'prediction_data': csv_path,
            'prediction_plot': plot_path
        }
        
        # Save result as JSON
        json_path = os.path.join(RESULTS_DIR, f"{ticker}_prediction_info.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        print(f"Generated predictions for {ticker}: Expected return over {days_ahead} days = {expected_return:.2f}%")
        return result
        
    except Exception as e:
        print(f"Error predicting future for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Create model trainer instance
    model_trainer = ModelTrainer(models_dir=MODELS_DIR)
    
    # Get all CSV files in the data directory
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('_Sorted.csv')]
    
    if not data_files:
        print("No data files found")
        return
    
    print(f"Found {len(data_files)} data files to process")
    
    # Overall results
    all_results = {}
    
    # Process each file
    for data_file in data_files:
        ticker = data_file.split('_')[0]
        print(f"\n=== Making predictions for {ticker} ===")
        
        # Load data
        data = load_data(os.path.join(DATA_DIR, data_file))
        if data is None:
            continue
            
        # Make predictions
        result = predict_future(model_trainer, data, ticker)
        
        if result:
            all_results[ticker] = result
    
    # Save overall results summary
    summary_path = os.path.join(RESULTS_DIR, "predictions_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary table with return expectations
    if all_results:
        # Sort by expected return
        sorted_results = sorted(all_results.values(), key=lambda x: x['expected_return_percent'], reverse=True)
        
        # Create summary table plot
        plt.figure(figsize=(12, 8))
        
        tickers = [r['ticker'] for r in sorted_results]
        returns = [r['expected_return_percent'] for r in sorted_results]
        
        # Plot as horizontal bar chart
        plt.barh(tickers, returns, color=['green' if r >= 0 else 'red' for r in returns])
        
        # Add labels
        plt.xlabel('Expected Return (%)')
        plt.ylabel('Stock')
        plt.title('30-Day Expected Returns by Stock')
        plt.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(returns):
            plt.text(v + (1 if v >= 0 else -1), i, f"{v:.2f}%", 
                     va='center', ha='left' if v >= 0 else 'right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "expected_returns_summary.png"))
        plt.close()
        
    # Print summary
    print(f"\n=== Prediction Summary ===")
    print(f"Generated predictions for {len(all_results)} stocks")
    print(f"Results saved to {RESULTS_DIR}")
    
    # Print top 3 and bottom 3 performers
    if all_results:
        sorted_tickers = sorted(all_results.keys(), 
                               key=lambda t: all_results[t]['expected_return_percent'],
                               reverse=True)
        
        print("\nTop performing stocks (expected return):")
        for i, ticker in enumerate(sorted_tickers[:3]):
            ret = all_results[ticker]['expected_return_percent']
            print(f"  {i+1}. {ticker}: {ret:.2f}%")
            
        print("\nBottom performing stocks (expected return):")
        for i, ticker in enumerate(sorted_tickers[-3:]):
            ret = all_results[ticker]['expected_return_percent']
            print(f"  {i+1}. {ticker}: {ret:.2f}%")

if __name__ == "__main__":
    main() 