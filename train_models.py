"""
Train and save stock prediction models for short, medium, and long timeframes.

This script:
1. Loads stock data from CSV files
2. Trains three different LSTM model architectures (simple, medium, complex)
3. For each architecture, trains models for three timeframes (short, medium, long)
4. Saves all 9 resulting models in .h5 format

Usage: python train_models.py
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
from src.model_trainer import ModelTrainer

# Configure directories
DATA_DIR = "./data/stocks"
MODELS_DIR = "./models"

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "short"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "medium"), exist_ok=True)
os.makedirs(os.path.join(MODELS_DIR, "long"), exist_ok=True)

# Define timeframes
TIMEFRAMES = {
    "short": {"sequence_length": 30, "prediction_horizons": [1, 3, 5]},
    "medium": {"sequence_length": 60, "prediction_horizons": [7, 14, 21]},
    "long": {"sequence_length": 90, "prediction_horizons": [30, 60, 90]}
}

# Define model architectures
def create_simple_model(input_shape, num_outputs):
    """Create a simple LSTM model with one layer"""
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.2),
        Dense(num_outputs)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_medium_model(input_shape, num_outputs):
    """Create a medium complexity LSTM model with two layers"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(num_outputs)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_complex_model(input_shape, num_outputs):
    """Create a more complex LSTM model with three layers and batch normalization"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_outputs)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

MODEL_ARCHITECTURES = {
    "simple": create_simple_model,
    "medium": create_medium_model,
    "complex": create_complex_model
}

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
        
        return data
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def create_sequences(data, seq_length, features, prediction_horizons):
    """Create sequences for training with multiple prediction horizons."""
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features].values)
    
    # Get max horizon for padding
    max_horizon = max(prediction_horizons)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - seq_length - max_horizon):
        # Input sequence
        X.append(scaled_data[i:i+seq_length])
        
        # Target values for each horizon
        targets = []
        for horizon in prediction_horizons:
            # Use Price column (assumed to be the first feature)
            price_idx = features.index('Price')
            targets.append(scaled_data[i+seq_length+horizon-1, price_idx])
        
        y.append(targets)
    
    return np.array(X), np.array(y), scaler

def train_model_for_ticker(data, ticker, timeframe, model_architecture):
    """Train a model for a specific ticker, timeframe, and architecture."""
    print(f"Training {model_architecture} model for {ticker} with {timeframe} timeframe")
    
    # Get ticker data
    ticker_data = data[data['ticker'] == ticker].copy()
    
    # Check if we have enough data
    tf_config = TIMEFRAMES[timeframe]
    seq_length = tf_config["sequence_length"]
    prediction_horizons = tf_config["prediction_horizons"]
    max_horizon = max(prediction_horizons)
    
    min_data_points = seq_length + max_horizon + 30  # Add buffer for training
    
    if len(ticker_data) < min_data_points:
        print(f"Insufficient data for {ticker}: {len(ticker_data)} < {min_data_points}")
        return None
    
    # Select features (basic feature set)
    base_features = ['Price', 'Open', 'High', 'Low']
    available_features = [f for f in base_features if f in ticker_data.columns]
    
    # Add volume and change if available
    if 'Vol.' in ticker_data.columns:
        available_features.append('Vol.')
    if 'Change %' in ticker_data.columns:
        available_features.append('Change %')
    
    # Add technical indicators if available
    technical_indicators = [col for col in ticker_data.columns if 
                           col.startswith('RSI') or 
                           col.startswith('MACD') or
                           col.startswith('SMA') or
                           col.startswith('EMA')]
    
    if technical_indicators:
        available_features.extend(technical_indicators)
    
    # Ensure we have enough features
    if len(available_features) < 3 or 'Price' not in available_features:
        print(f"Not enough features for {ticker}, or missing Price column")
        return None
    
    # Create sequences
    X, y, scaler = create_sequences(ticker_data, seq_length, available_features, prediction_horizons)
    
    if len(X) == 0:
        print(f"Failed to create sequences for {ticker}")
        return None
    
    # Split into train and validation sets (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create model using the specified architecture
    input_shape = (X_train.shape[1], X_train.shape[2])
    model_fn = MODEL_ARCHITECTURES[model_architecture]
    model = model_fn(input_shape, len(prediction_horizons))
    
    # Train with early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Determine batch size based on data size
    batch_size = min(32, len(X_train) // 4) if len(X_train) > 4 else 1
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save model and related artifacts
    models_subdir = os.path.join(MODELS_DIR, timeframe)
    os.makedirs(models_subdir, exist_ok=True)
    
    # Filename pattern: ticker_architecture_timeframe.h5
    model_path = os.path.join(models_subdir, f"{ticker}_{model_architecture}_{timeframe}.h5")
    model.save(model_path)
    
    # Save scaler
    scaler_path = os.path.join(models_subdir, f"{ticker}_{model_architecture}_{timeframe}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save features
    features_path = os.path.join(models_subdir, f"{ticker}_{model_architecture}_{timeframe}_features.pkl")
    with open(features_path, 'wb') as f:
        pickle.dump(available_features, f)
    
    # Save metadata
    metadata = {
        'ticker': ticker,
        'architecture': model_architecture,
        'timeframe': timeframe,
        'sequence_length': seq_length,
        'prediction_horizons': prediction_horizons,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'features': available_features,
        'epochs': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'training_samples': len(X_train),
        'validation_samples': len(X_val)
    }
    
    metadata_path = os.path.join(models_subdir, f"{ticker}_{model_architecture}_{timeframe}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{ticker} - {model_architecture.capitalize()} Model - {timeframe.capitalize()} Timeframe')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plot_path = os.path.join(models_subdir, f"{ticker}_{model_architecture}_{timeframe}_training.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Successfully trained and saved {model_architecture} model for {ticker} ({timeframe} timeframe)")
    return {
        'model': model,
        'scaler': scaler,
        'features': available_features,
        'metadata': metadata
    }

def main():
    # Get all CSV files in the data directory
    data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('_Sorted.csv')]
    
    if not data_files:
        print("No data files found")
        return
    
    print(f"Found {len(data_files)} data files to process")
    
    # Results tracking
    results = {
        timeframe: {model: [] for model in MODEL_ARCHITECTURES} 
        for timeframe in TIMEFRAMES
    }
    
    # Process each file
    for data_file in data_files:
        ticker = data_file.split('_')[0]
        print(f"\n=== Processing {ticker} ===")
        
        # Load data
        data = load_data(os.path.join(DATA_DIR, data_file))
        if data is None or data.empty:
            print(f"Skipping {ticker} - failed to load data")
            continue
        
        # Train models for different timeframes and architectures
        for timeframe in TIMEFRAMES:
            for architecture in MODEL_ARCHITECTURES:
                try:
                    model_result = train_model_for_ticker(data, ticker, timeframe, architecture)
                    if model_result:
                        results[timeframe][architecture].append(ticker)
                except Exception as e:
                    print(f"Error training {architecture} model for {ticker} ({timeframe}): {str(e)}")
    
    # Print summary
    print("\n=== Training Summary ===")
    for timeframe in TIMEFRAMES:
        print(f"\n{timeframe.upper()} TIMEFRAME:")
        for architecture in MODEL_ARCHITECTURES:
            tickers = results[timeframe][architecture]
            print(f"  {architecture.capitalize()} model: {len(tickers)} trained successfully")
            if tickers:
                print(f"    Tickers: {', '.join(tickers[:5])}" + (f" and {len(tickers)-5} more" if len(tickers) > 5 else ""))
    
    # Save summary
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': results,
        'timeframes': {tf: config for tf, config in TIMEFRAMES.items()},
        'total_models_trained': sum(len(tickers) for tf in results for arch, tickers in results[tf].items())
    }
    
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining summary saved to {summary_path}")

if __name__ == "__main__":
    main() 