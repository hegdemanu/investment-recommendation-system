"""
Validate model performance with train-test split and retrain on full dataset.
This script:
1. Loads data and splits into training (8 months) and testing (2 months) periods
2. Trains LSTM models on the training data
3. Evaluates model performance on test data
4. Then trains on the full dataset and saves models
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.model_trainer import ModelTrainer

# Configure output directories
MODELS_DIR = "./models"
RESULTS_DIR = "./results"
DATA_DIR = "./data/stocks"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

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
        print(f"Data types: {data.dtypes}")
        return data
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def split_train_test(data, train_months=8):
    """Split data into training and testing sets."""
    # Sort by date
    data = data.sort_values('Date')
    
    # Find the date that separates training and testing
    latest_date = data['Date'].max()
    earliest_date = data['Date'].min()
    
    # Calculate duration and split point
    total_days = (latest_date - earliest_date).days
    train_days = int(total_days * (train_months / 10))  # 8 months out of 10
    
    split_date = earliest_date + timedelta(days=train_days)
    
    print(f"Data range: {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
    print(f"Split date: {split_date.strftime('%Y-%m-%d')} ({train_months} months training)")
    
    # Split data
    train_data = data[data['Date'] <= split_date].copy()
    test_data = data[data['Date'] > split_date].copy()
    
    print(f"Training data: {len(train_data)} rows ({len(train_data)/len(data):.1%})")
    print(f"Testing data: {len(test_data)} rows ({len(test_data)/len(data):.1%})")
    
    return train_data, test_data

def calculate_metrics(true_values, predictions):
    """Calculate performance metrics."""
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }

def evaluate_model(model_trainer, models, test_data, ticker):
    """Evaluate model performance on test data."""
    if ticker not in models['models']:
        print(f"No model found for ticker {ticker}")
        return None
    
    # Get model components
    model = models['models'][ticker]['model']
    scaler = models['models'][ticker]['scaler']
    features = models['models'][ticker]['features']
    
    # Filter test data for this ticker
    ticker_test_data = test_data[test_data['ticker'] == ticker].copy()
    
    if len(ticker_test_data) < model_trainer.sequence_length + 1:
        print(f"Not enough test data for {ticker}")
        return None
        
    # Prepare sequences for prediction
    last_sequence, used_features = model_trainer.prepare_sequence_for_prediction(
        ticker_test_data.iloc[:model_trainer.sequence_length], 
        features, 
        scaler
    )
    
    if last_sequence is None:
        print(f"Failed to prepare sequence for {ticker}")
        return None
    
    # Get prediction horizons info
    horizons_path = os.path.join(MODELS_DIR, f"{ticker}_horizons.pkl")
    if os.path.exists(horizons_path):
        with open(horizons_path, 'rb') as f:
            horizons_info = pickle.load(f)
        horizons = horizons_info['prediction_horizons']
        best_horizon = horizons_info['best_horizon']
    else:
        horizons = [1, 3, 7, 14, 30]
        best_horizon = 1
    
    # Make predictions for the test period
    future_days = len(ticker_test_data) - model_trainer.sequence_length
    predictions = model_trainer.predict_future(model, last_sequence, scaler, used_features, days=future_days)
    
    if len(predictions) == 0:
        print(f"Failed to generate predictions for {ticker}")
        return None
    
    # Get actual prices from test data
    actual_prices = ticker_test_data['Price'].values[model_trainer.sequence_length:]
    
    # Ensure same length for comparison
    min_len = min(len(actual_prices), len(predictions))
    actual_prices = actual_prices[:min_len]
    predictions = predictions[:min_len]
    
    # Calculate metrics
    metrics = calculate_metrics(actual_prices, predictions)
    print(f"Performance metrics for {ticker}:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    # Create plots
    plt.figure(figsize=(12, 6))
    plt.plot(ticker_test_data['Date'].values[model_trainer.sequence_length:model_trainer.sequence_length+min_len], 
             actual_prices, label='Actual', color='blue')
    plt.plot(ticker_test_data['Date'].values[model_trainer.sequence_length:model_trainer.sequence_length+min_len], 
             predictions, label='Predicted', color='red', linestyle='--')
    plt.title(f'LSTM Model Predictions vs Actual for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, "plots", f"{ticker}_prediction_test.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Plot prediction error
    plt.figure(figsize=(12, 6))
    errors = actual_prices - predictions
    plt.plot(ticker_test_data['Date'].values[model_trainer.sequence_length:model_trainer.sequence_length+min_len], 
             errors, color='green')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f'Prediction Error for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Error (Actual - Predicted)')
    plt.grid(True)
    
    # Save error plot
    error_plot_path = os.path.join(RESULTS_DIR, "plots", f"{ticker}_prediction_error.png")
    plt.savefig(error_plot_path)
    plt.close()
    
    # Return results
    result = {
        'ticker': ticker,
        'metrics': metrics,
        'data_points': min_len,
        'plots': {
            'prediction': plot_path,
            'error': error_plot_path
        },
        'best_horizon': best_horizon
    }
    
    # Save metrics to file
    metrics_path = os.path.join(RESULTS_DIR, "metrics", f"{ticker}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

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
        print(f"\n=== Processing {ticker} ===")
        
        # Load data
        data = load_data(os.path.join(DATA_DIR, data_file))
        if data is None:
            continue
            
        # Check data types to ensure all numeric columns are properly converted
        for col in ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                print(f"Warning: Column {col} is not numeric. Converting to numeric.")
                data[col] = pd.to_numeric(data[col], errors='coerce')
                data[col] = data[col].fillna(data[col].mean() if data[col].mean() else 0)
        
        # Split data
        train_data, test_data = split_train_test(data, train_months=8)
        
        try:
            # Train model on training data
            print(f"Training model for {ticker} on training data...")
            models = model_trainer.train_lstm_models(train_data)
            
            if ticker not in models['models']:
                print(f"Failed to train model for {ticker}")
                continue
            
            # Evaluate model on test data
            print(f"Evaluating model for {ticker} on test data...")
            result = evaluate_model(model_trainer, models, test_data, ticker)
            
            if result:
                all_results[ticker] = result
            
            # Now train on the full dataset for future use
            print(f"Training model for {ticker} on full dataset...")
            models_full = model_trainer.train_lstm_models(data)
            
            if ticker in models_full['models']:
                print(f"Successfully trained model for {ticker} on full dataset")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue
    
    # Save overall results
    summary_path = os.path.join(RESULTS_DIR, "validation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n=== Validation Summary ===")
    print(f"Processed {len(data_files)} stocks")
    print(f"Successfully validated {len(all_results)} models")
    
    # Calculate average metrics
    if all_results:
        avg_metrics = {
            'rmse': np.mean([r['metrics']['rmse'] for r in all_results.values()]),
            'mae': np.mean([r['metrics']['mae'] for r in all_results.values()]),
            'r2': np.mean([r['metrics']['r2'] for r in all_results.values()]),
            'mape': np.mean([r['metrics']['mape'] for r in all_results.values()])
        }
        
        print("\nAverage metrics across all models:")
        for metric, value in avg_metrics.items():
            print(f"  Avg {metric.upper()}: {value:.4f}")
    
    print(f"\nDetailed results saved to {summary_path}")
    print(f"Plots saved to {os.path.join(RESULTS_DIR, 'plots')}")

if __name__ == "__main__":
    main() 