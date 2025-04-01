#!/usr/bin/env python
# coding: utf-8

"""
Run TimeframeInvestmentAnalyzer for Investment Recommendation System
This script runs the TimeframeInvestmentAnalyzer using the enhanced data and ModelBuilder fix.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from datetime import datetime, timedelta
import json

# Import ModelBuilder from analyzer_fix
from models.analyzer_fix import ModelBuilder

# Suppress warnings
warnings.filterwarnings('ignore')

print("=== Investment Recommendation System - TimeframeInvestmentAnalyzer ===")

# Check if enhanced_data.csv exists
if not os.path.exists('data/enhanced_data.csv'):
    print("Error: Enhanced data file (data/enhanced_data.csv) not found.")
    print("Please run process_data.py first to create the enhanced data.")
    exit(1)

# Load enhanced data
print("Loading enhanced data...")
enhanced_data = pd.read_csv('data/enhanced_data.csv')

# Convert Date column to datetime if needed
if 'Date' in enhanced_data.columns and not pd.api.types.is_datetime64_any_dtype(enhanced_data['Date']):
    enhanced_data['Date'] = pd.to_datetime(enhanced_data['Date'])

# Get ticker from data
if 'ticker' not in enhanced_data.columns:
    print("Error: No ticker column found in data")
    exit(1)

ticker = enhanced_data['ticker'].iloc[0]
print(f"Analyzing investment for ticker: {ticker}")

# Create TimeframeInvestmentAnalyzer class
class TimeframeInvestmentAnalyzer:
    """Class for analyzing investments across different timeframes"""
    
    def __init__(self, models_dir='models', results_dir='results'):
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Ensure directories exist
        for directory in [
            f'{models_dir}/short_term', 
            f'{models_dir}/medium_term', 
            f'{models_dir}/long_term',
            f'{results_dir}/short_term',
            f'{results_dir}/medium_term', 
            f'{results_dir}/long_term'
        ]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
    def get_timeframe_params(self, timeframe):
        """Get parameters for different timeframes"""
        params = {
            'short': {
                'seq_length': 10,
                'forecast_days': 7,
                'features': ['Price', 'RSI_14', 'MACD', 'Volatility_5d', 'Price_1d_change'],
                'model_file': f"{self.models_dir}/short_term/lstm_{{}}.h5",
                'results_dir': f"{self.results_dir}/short_term"
            },
            'medium': {
                'seq_length': 20,
                'forecast_days': 30,
                'features': ['Price', 'SMA_20', 'EMA_20', 'RSI_14', 'Volatility_20d', 'MACD'],
                'model_file': f"{self.models_dir}/medium_term/lstm_{{}}.h5",
                'results_dir': f"{self.results_dir}/medium_term"
            },
            'long': {
                'seq_length': 50,
                'forecast_days': 90,
                'features': ['Price', 'SMA_50', 'RSI_14', 'Volatility_20d', 'Price_20d_change', 'MACD'],
                'model_file': f"{self.models_dir}/long_term/lstm_{{}}.h5",
                'results_dir': f"{self.results_dir}/long_term"
            }
        }
        
        return params.get(timeframe, None)
                
    def predict_all_timeframes(self, data, ticker, try_load_existing=True):
        """Train models and make predictions for all timeframes"""
        results = {}
        
        for timeframe in ['short', 'medium', 'long']:
            print(f"\n===== {timeframe.capitalize()}-Term Analysis for {ticker} =====")
            
            params = self.get_timeframe_params(timeframe)
            if params is None:
                print(f"Invalid timeframe: {timeframe}")
                continue
                
            model_file = params['model_file'].format(ticker)
            
            if try_load_existing and os.path.exists(model_file):
                # Try to load existing model
                print(f"Loading existing {timeframe}-term model for {ticker}")
                try:
                    model = load_model(model_file)
                    
                    # We also need to load the scaler
                    scaler_file = model_file.replace('.h5', '_scaler.pkl')
                    features_file = model_file.replace('.h5', '_features.json')
                    
                    if os.path.exists(scaler_file) and os.path.exists(features_file):
                        import pickle
                        with open(scaler_file, 'rb') as f:
                            scaler = pickle.load(f)
                            
                        with open(features_file, 'r') as f:
                            features = json.load(f)
                            
                        print(f"Loaded {timeframe}-term model and scaler")
                    else:
                        print(f"Scaler or features not found, retraining {timeframe}-term model")
                        model, scaler, features = self.train_timeframe_model(data, ticker, timeframe)
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    model, scaler, features = self.train_timeframe_model(data, ticker, timeframe)
            else:
                # Train new model
                print(f"Training new {timeframe}-term model for {ticker}")
                model, scaler, features = self.train_timeframe_model(data, ticker, timeframe)
                
            if model is None:
                print(f"Failed to train {timeframe}-term model")
                continue
                
            # Generate predictions
            predictions = self.generate_predictions(data, ticker, model, scaler, features, timeframe)
            
            if predictions is None or predictions.empty:
                print(f"Failed to generate {timeframe}-term predictions")
                continue
                
            # Store results
            results[timeframe] = {
                'model': model,
                'scaler': scaler,
                'features': features,
                'predictions': predictions
            }
            
            # Save predictions
            if not os.path.exists(params['results_dir']):
                os.makedirs(params['results_dir'])
                
            predictions.to_csv(f"{params['results_dir']}/{ticker}_predictions.csv", index=False)
            
            # Save model if newly trained
            if not try_load_existing or not os.path.exists(model_file):
                model.save(model_file)
                
                # Save scaler and features
                import pickle
                with open(model_file.replace('.h5', '_scaler.pkl'), 'wb') as f:
                    pickle.dump(scaler, f)
                    
                with open(model_file.replace('.h5', '_features.json'), 'w') as f:
                    json.dump(features, f)
                    
                print(f"Saved {timeframe}-term model and scaler")
                
        return results
    
    def train_timeframe_model(self, data, ticker, timeframe):
        """Train an LSTM model for a specific timeframe"""
        params = self.get_timeframe_params(timeframe)
        if params is None:
            print(f"Invalid timeframe: {timeframe}")
            return None, None, None
            
        # Get the data for the ticker
        if 'ticker' in data.columns:
            ticker_data = data[data['ticker'] == ticker].copy()
        else:
            ticker_data = data.copy()
            
        # Check if we have enough data
        if len(ticker_data) < params['seq_length'] * 2:
            print(f"Insufficient data for {timeframe}-term analysis: {len(ticker_data)} rows")
            return None, None, None
            
        # Select features
        all_features = params['features']
        
        # Check for missing features
        missing_features = [f for f in all_features if f not in ticker_data.columns]
        if missing_features:
            print(f"Warning: Missing features for {timeframe}-term analysis: {missing_features}")
            
            # Use available features
            features = [f for f in all_features if f in ticker_data.columns]
            if not features:
                print(f"No valid features available for {timeframe}-term analysis")
                return None, None, None
        else:
            features = all_features
            
        print(f"Selected features for {timeframe}-term analysis: {features[:5]}")
        if len(features) > 5:
            print(f"  ...plus {len(features) - 5} additional features")
            
        # Create model builder
        model_builder = ModelBuilder(models_dir=f"{self.models_dir}/{timeframe}_term")
        
        # Prepare data for LSTM
        seq_length = params['seq_length']
        data_sequences, scaler, features = model_builder.prepare_lstm_data(
            ticker_data, ticker, seq_length=seq_length
        )
        
        if data_sequences is None:
            print(f"Error preparing data sequences for {timeframe}-term {ticker} model")
            return None, None, None
            
        # Unpack sequences
        X_train, y_train, X_val, y_val, _, _ = data_sequences
        
        # Build LSTM model
        lstm_units = 50
        dropout_rate = 0.2
        
        model = Sequential()
        model.add(LSTM(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Create callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        print(f"Model trained for {len(history.history['loss'])} epochs")
        
        return model, scaler, features
    
    def generate_predictions(self, data, ticker, model, scaler, features, timeframe):
        """Generate predictions for the given timeframe"""
        params = self.get_timeframe_params(timeframe)
        if params is None:
            print(f"Invalid timeframe: {timeframe}")
            return None
            
        forecast_days = params['forecast_days']
        seq_length = params['seq_length']
        
        print(f"Generating {forecast_days}-day forecast for {ticker} ({timeframe}-term)")
        
        try:
            # Generate predictions
            # Get latest data for prediction
            latest_data = data[data['ticker'] == ticker].copy() if 'ticker' in data.columns else data.copy()
            latest_data = latest_data.sort_values('Date')
            
            # Make sure we have the features needed
            missing_features = [f for f in features if f not in latest_data.columns]
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                return None
                
            # Get the last sequence for prediction
            last_sequence = latest_data[features].tail(seq_length).values
            
            # Scale the data
            last_sequence_scaled = scaler.transform(last_sequence)
            input_seq = np.array([last_sequence_scaled])
            
            # Make predictions for the forecast period
            future_dates = []
            future_prices = []
            
            last_date = latest_data['Date'].iloc[-1]
            current_seq = input_seq.copy()
            
            for i in range(forecast_days):
                # Predict next price
                pred_scaled = model.predict(current_seq, verbose=0)[0][0]
                
                # Create dummy row for inverse scaling
                dummy = np.zeros((1, len(features)))
                dummy[0, 0] = pred_scaled
                
                # Get actual price prediction
                pred_price = scaler.inverse_transform(dummy)[0][0]
                future_prices.append(pred_price)
                
                # Calculate next date (skip weekends)
                next_date = last_date + timedelta(days=i+1)
                while next_date.weekday() > 4:  # Skip Saturday and Sunday
                    next_date = next_date + timedelta(days=1)
                    
                future_dates.append(next_date)
                
                # Update the sequence for next prediction (slide window)
                new_row = np.copy(current_seq[0, 1:, :])
                new_row = np.append(new_row, [last_sequence_scaled[-1]], axis=0)
                new_row[-1, 0] = pred_scaled
                current_seq = np.array([new_row])
                
            # Create dataframe with predictions
            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Price': future_prices,
                'ticker': ticker,
                'timeframe': timeframe
            })
            
            return predictions_df
            
        except Exception as e:
            print(f"Error generating predictions: {str(e)}")
            return None
    
    def generate_timeframe_report(self, results, ticker, risk_profile='moderate'):
        """Generate an investment recommendation report based on timeframe predictions"""
        if not results:
            print("No results available for report generation")
            return
            
        print(f"\n===== Investment Report for {ticker} ({risk_profile} risk profile) =====")
        
        # Combine predictions from all timeframes
        all_predictions = []
        
        for timeframe, data in results.items():
            predictions = data['predictions']
            if predictions is not None and not predictions.empty:
                all_predictions.append(predictions)
                
        if not all_predictions:
            print("No valid predictions available for report")
            return
            
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        
        # Compute expected returns for each timeframe
        returns = {}
        volatility = {}
        
        for timeframe in ['short', 'medium', 'long']:
            if timeframe in results:
                predictions = results[timeframe]['predictions']
                if predictions is not None and not predictions.empty:
                    # Calculate return
                    initial_price = predictions['Predicted_Price'].iloc[0]
                    final_price = predictions['Predicted_Price'].iloc[-1]
                    timeframe_return = (final_price - initial_price) / initial_price * 100
                    
                    # Calculate volatility
                    daily_returns = predictions['Predicted_Price'].pct_change().std() * 100
                    
                    returns[timeframe] = timeframe_return
                    volatility[timeframe] = daily_returns
        
        # Generate recommendation based on risk profile
        recommendation = "HOLD"  # Default
        
        if risk_profile == 'conservative':
            # Conservative investors care more about lower volatility
            if 'long' in returns and returns['long'] > 0 and volatility.get('long', float('inf')) < 2.0:
                recommendation = "BUY"
                timeframe = "long-term"
            elif 'medium' in returns and returns['medium'] > 0 and volatility.get('medium', float('inf')) < 1.5:
                recommendation = "BUY"
                timeframe = "medium-term"
            elif all(r < 0 for r in returns.values()):
                recommendation = "SELL"
                timeframe = "all timeframes"
                
        elif risk_profile == 'aggressive':
            # Aggressive investors care more about higher returns
            if 'short' in returns and returns['short'] > 3.0:
                recommendation = "BUY"
                timeframe = "short-term"
            elif 'medium' in returns and returns['medium'] > 7.0:
                recommendation = "BUY"
                timeframe = "medium-term"
            elif all(r < -5 for r in returns.values()):
                recommendation = "SELL"
                timeframe = "all timeframes"
                
        else:  # moderate
            # Balanced approach
            if 'medium' in returns and returns['medium'] > 5.0 and volatility.get('medium', float('inf')) < 2.5:
                recommendation = "BUY"
                timeframe = "medium-term"
            elif 'long' in returns and returns['long'] > 10.0:
                recommendation = "BUY"
                timeframe = "long-term"
            elif 'short' in returns and returns['short'] > 2.0 and volatility.get('short', float('inf')) < 2.0:
                recommendation = "BUY"
                timeframe = "short-term"
            elif all(r < -3 for r in returns.values()):
                recommendation = "SELL"
                timeframe = "all timeframes"
        
        # Print recommendation
        print(f"\nRecommendation for {ticker}: {recommendation}")
        
        # Print expected returns
        print("\nExpected Returns:")
        for timeframe, ret in returns.items():
            if timeframe == 'short':
                period = "7 days"
            elif timeframe == 'medium':
                period = "30 days"
            else:
                period = "90 days"
                
            print(f"  {timeframe.capitalize()}-term ({period}): {ret:.2f}%")
            
        # Print volatility
        print("\nExpected Volatility:")
        for timeframe, vol in volatility.items():
            print(f"  {timeframe.capitalize()}-term: {vol:.2f}%")
            
        # Save report
        report_dir = "reports"
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
            
        with open(f"{report_dir}/{ticker}_{risk_profile}_report.txt", 'w') as f:
            f.write(f"Investment Report for {ticker} ({risk_profile} risk profile)\n")
            f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Recommendation: {recommendation}\n\n")
            
            f.write("Expected Returns:\n")
            for timeframe, ret in returns.items():
                if timeframe == 'short':
                    period = "7 days"
                elif timeframe == 'medium':
                    period = "30 days"
                else:
                    period = "90 days"
                    
                f.write(f"  {timeframe.capitalize()}-term ({period}): {ret:.2f}%\n")
                
            f.write("\nExpected Volatility:\n")
            for timeframe, vol in volatility.items():
                f.write(f"  {timeframe.capitalize()}-term: {vol:.2f}%\n")
                
        print(f"\nReport saved to {report_dir}/{ticker}_{risk_profile}_report.txt")

# Create and run the analyzer
timeframe_analyzer = TimeframeInvestmentAnalyzer()
results = timeframe_analyzer.predict_all_timeframes(enhanced_data, ticker)

if results:
    # Generate reports for different risk profiles
    for risk_profile in ['conservative', 'moderate', 'aggressive']:
        timeframe_analyzer.generate_timeframe_report(results, ticker, risk_profile)
else:
    print("Failed to generate predictions. Please check the data and try again.") 