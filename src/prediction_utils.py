"""
Utility functions for generating predictions and analyzing stock data.
These functions are used for prediction, risk assessment, and recommendation generation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def process_volume(vol_str):
    """Process volume string with K/M/B suffixes to numeric value."""
    if pd.isna(vol_str) or vol_str == '-':
        return np.nan
    
    if isinstance(vol_str, (int, float)):
        return float(vol_str)
        
    vol_str = str(vol_str).strip()
    if vol_str.endswith('K'):
        return float(vol_str[:-1]) * 1000
    elif vol_str.endswith('M'):
        return float(vol_str[:-1]) * 1000000
    elif vol_str.endswith('B'):
        return float(vol_str[:-1]) * 1000000000
    elif vol_str:
        return float(vol_str)
    return np.nan

def process_percentage(value):
    """Process percentage string to numeric value."""
    if pd.isna(value) or value == '-':
        return np.nan
        
    if isinstance(value, (int, float)):
        return float(value)
        
    value = str(value).strip()
    if value.endswith('%'):
        return float(value[:-1]) / 100
    elif value:
        return float(value)
    return np.nan

def prepare_sequence(data, features, scaler, seq_length=60):
    """Prepare sequence data for LSTM model."""
    if len(data) < seq_length:
        return None, None
    
    # Get selected features and scale
    data_subset = data[features].copy()
    data_scaled = scaler.transform(data_subset)
    
    # Create sequence
    sequence = np.array([data_scaled[-seq_length:]])
    return sequence, features

def predict_future(model, last_sequence, scaler, available_features, days=30):
    """Generate predictions for future days."""
    predictions = []
    current_sequence = last_sequence.copy()
    
    # For each day we want to predict
    for i in range(days):
        # Make prediction
        pred = model.predict(current_sequence)
        
        # Get the predicted value
        predicted_scaled = pred[0][0]
        
        # Create a dummy row with the predicted value
        dummy_row = np.zeros((1, len(available_features)))
        
        # Find the index of the price column to set the predicted value
        try:
            price_idx = available_features.index('Price')
            dummy_row[0, price_idx] = predicted_scaled
            
            # Inverse transform to get the actual price value
            predicted_price = scaler.inverse_transform(dummy_row)[0, price_idx]
            predictions.append(predicted_price)
            
            # Update sequence for next prediction
            # Remove first element and append the prediction
            current_sequence = np.append(current_sequence[:, 1:, :], 
                                        np.array([[[predicted_scaled]]]), 
                                        axis=1)
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            break
            
    return predictions

def backtracking_analysis(data, model, scaler, features, periods):
    """Perform backtracking analysis on historical data."""
    seq_length = model.layers[0].input_shape[1]
    
    results = {}
    
    # Set periods to analyze
    if not periods:
        periods = {
            "1_month": 30,
            "3_months": 90,
            "6_months": 180
        }
        
    data_sorted = data.sort_values('Date')
    
    for period_name, days in periods.items():
        print(f"Analyzing {period_name} ({days} days)...")
        
        # Skip if not enough data
        if len(data_sorted) <= days + seq_length:
            print(f"Not enough data for {period_name} analysis")
            continue
            
        # Split data
        train_data = data_sorted.iloc[:-days].copy()
        test_data = data_sorted.iloc[-days:].copy()
        
        # Prepare sequence
        sequence, used_features = prepare_sequence(train_data, features, scaler, seq_length)
        
        if sequence is None:
            print(f"Could not create sequence for {period_name}")
            continue
            
        # Generate predictions
        predictions = predict_future(model, sequence, scaler, used_features, days=days)
        
        # Get actual prices
        actual_prices = test_data['Price'].values
        
        # Calculate metrics
        mse = np.mean((actual_prices - predictions) ** 2)
        mae = np.mean(np.abs(actual_prices - predictions))
        mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
        
        results[period_name] = {
            'predictions': predictions,
            'actual': actual_prices.tolist(),
            'dates': test_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'mape': float(mape)
            }
        }
        
    return results

def predict_multi_timeframe(model, last_sequence, scaler, available_features):
    """Generate predictions for multiple time frames."""
    # Define time frames
    time_frames = {
        'short_term': 7,    # 1 week
        'medium_term': 30,  # 1 month
        'long_term': 90     # 3 months
    }
    
    results = {}
    
    # Generate predictions for each time frame
    for time_frame_name, days in time_frames.items():
        predictions = predict_future(model, last_sequence, scaler, available_features, days=days)
        results[time_frame_name] = predictions
        
    return results

def generate_risk_based_recommendations(predictions_multi_tf, last_price, risk_appetite='moderate'):
    """Generate investment recommendations based on risk appetite."""
    # Extract predictions for different time frames
    short_term = predictions_multi_tf.get('short_term', [])
    medium_term = predictions_multi_tf.get('medium_term', [])
    long_term = predictions_multi_tf.get('long_term', [])
    
    # Calculate expected returns
    returns = {}
    
    if short_term and len(short_term) > 0:
        returns['short_term'] = (short_term[-1] / last_price - 1) * 100
        
    if medium_term and len(medium_term) > 0:
        returns['medium_term'] = (medium_term[-1] / last_price - 1) * 100
        
    if long_term and len(long_term) > 0:
        returns['long_term'] = (long_term[-1] / last_price - 1) * 100
    
    # Calculate volatility
    volatility = {}
    
    if short_term and len(short_term) > 1:
        short_term_returns = np.diff(short_term) / short_term[:-1]
        volatility['short_term'] = np.std(short_term_returns) * 100
        
    if medium_term and len(medium_term) > 1:
        medium_term_returns = np.diff(medium_term) / medium_term[:-1]
        volatility['medium_term'] = np.std(medium_term_returns) * 100
        
    if long_term and len(long_term) > 1:
        long_term_returns = np.diff(long_term) / long_term[:-1]
        volatility['long_term'] = np.std(long_term_returns) * 100
    
    # Generate recommendation based on risk appetite
    recommendation = {
        'risk_appetite': risk_appetite,
        'returns': returns,
        'volatility': volatility,
        'recommendation': {}
    }
    
    # Decision logic based on risk appetite
    if risk_appetite.lower() == 'conservative':
        # Conservative investors prioritize stability
        if 'short_term' in returns and returns['short_term'] > 2:
            recommendation['recommendation']['action'] = 'BUY'
            recommendation['recommendation']['timeframe'] = 'short_term'
            recommendation['recommendation']['reason'] = 'Positive short-term outlook with acceptable risk'
        elif 'medium_term' in returns and returns['medium_term'] < -5:
            recommendation['recommendation']['action'] = 'SELL'
            recommendation['recommendation']['timeframe'] = 'medium_term'
            recommendation['recommendation']['reason'] = 'Significant medium-term downside risk'
        else:
            recommendation['recommendation']['action'] = 'HOLD'
            recommendation['recommendation']['timeframe'] = 'short_term'
            recommendation['recommendation']['reason'] = 'No clear short-term opportunity with acceptable risk'
            
    elif risk_appetite.lower() == 'aggressive':
        # Aggressive investors prioritize returns
        if 'long_term' in returns and returns['long_term'] > 10:
            recommendation['recommendation']['action'] = 'BUY'
            recommendation['recommendation']['timeframe'] = 'long_term'
            recommendation['recommendation']['reason'] = 'Strong long-term growth potential'
        elif 'medium_term' in returns and returns['medium_term'] > 5:
            recommendation['recommendation']['action'] = 'BUY'
            recommendation['recommendation']['timeframe'] = 'medium_term'
            recommendation['recommendation']['reason'] = 'Good medium-term growth potential'
        elif 'medium_term' in returns and returns['medium_term'] < -10:
            recommendation['recommendation']['action'] = 'SELL'
            recommendation['recommendation']['timeframe'] = 'medium_term'
            recommendation['recommendation']['reason'] = 'Significant medium-term downside'
        else:
            recommendation['recommendation']['action'] = 'HOLD'
            recommendation['recommendation']['timeframe'] = 'medium_term'
            recommendation['recommendation']['reason'] = 'No clear opportunity with sufficient return potential'
            
    else:  # moderate
        # Moderate investors balance risk and return
        if 'medium_term' in returns and returns['medium_term'] > 5:
            recommendation['recommendation']['action'] = 'BUY'
            recommendation['recommendation']['timeframe'] = 'medium_term'
            recommendation['recommendation']['reason'] = 'Good medium-term growth outlook'
        elif 'short_term' in returns and returns['short_term'] < -5:
            recommendation['recommendation']['action'] = 'SELL'
            recommendation['recommendation']['timeframe'] = 'short_term'
            recommendation['recommendation']['reason'] = 'Significant short-term downside risk'
        else:
            recommendation['recommendation']['action'] = 'HOLD'
            recommendation['recommendation']['timeframe'] = 'medium_term'
            recommendation['recommendation']['reason'] = 'No clear opportunity with balanced risk-return'
    
    return recommendation

def calculate_peg_ratio(price, eps, eps_growth):
    """Calculate the Price/Earnings to Growth (PEG) ratio."""
    if eps == 0 or eps_growth == 0:
        return None
        
    pe_ratio = price / eps
    peg_ratio = pe_ratio / eps_growth
    
    return peg_ratio

def evaluate_peg_ratio(peg_ratio):
    """Evaluate the PEG ratio and provide an interpretation."""
    if peg_ratio is None:
        return {
            'evaluation': 'UNKNOWN',
            'description': 'Cannot calculate PEG ratio with the provided data'
        }
        
    if peg_ratio < 0:
        return {
            'evaluation': 'CAUTION',
            'description': 'Negative PEG ratio indicates declining earnings growth'
        }
        
    if peg_ratio < 0.5:
        return {
            'evaluation': 'STRONGLY_UNDERVALUED',
            'description': 'PEG ratio below 0.5 suggests the stock is significantly undervalued'
        }
        
    if peg_ratio < 1.0:
        return {
            'evaluation': 'UNDERVALUED',
            'description': 'PEG ratio below 1.0 suggests the stock may be undervalued'
        }
        
    if peg_ratio < 1.5:
        return {
            'evaluation': 'FAIR_VALUE',
            'description': 'PEG ratio between 1.0 and 1.5 suggests the stock is reasonably valued'
        }
        
    if peg_ratio < 2.0:
        return {
            'evaluation': 'OVERVALUED',
            'description': 'PEG ratio between 1.5 and 2.0 suggests the stock may be overvalued'
        }
        
    return {
        'evaluation': 'STRONGLY_OVERVALUED',
        'description': 'PEG ratio above 2.0 suggests the stock is significantly overvalued'
    } 