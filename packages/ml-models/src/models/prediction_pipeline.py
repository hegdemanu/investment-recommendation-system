#!/usr/bin/env python3
"""
Prediction Pipeline for Investment Recommendation System
Implements ensemble predictions using LSTM, ARIMA+GARCH, and Prophet models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

import tensorflow as tf
from prophet import Prophet
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self, models: Dict = None, config: Dict = None):
        self.models = models or {}
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict:
        return {
            'short_term': {
                'lstm': 0.8,
                'arima_garch': 0.1,
                'prophet': 0.1
            },
            'medium_term': {
                'lstm': 0.4,
                'arima_garch': 0.4,
                'prophet': 0.2
            },
            'long_term': {
                'lstm': 0.2,
                'arima_garch': 0.3,
                'prophet': 0.5
            }
        }
    
    def get_weights(self, horizon: int) -> Dict[str, float]:
        """Get model weights based on prediction horizon"""
        if horizon <= 15:
            return self.config['short_term']
        elif horizon <= 30:
            return self.config['medium_term']
        else:
            return self.config['long_term']
    
    def predict_lstm(self, data: pd.DataFrame, horizon: int, scaler: MinMaxScaler) -> np.ndarray:
        """Generate LSTM predictions"""
        if 'lstm' not in self.models:
            raise ValueError("LSTM model not found")
            
        # Prepare sequence
        sequence = scaler.transform(data[['Close']].tail(60))
        sequence = sequence.reshape((1, 60, 1))
        
        # Generate predictions
        predictions = []
        current_sequence = sequence
        
        for _ in range(horizon):
            # Predict next value
            next_pred = self.models['lstm'].predict(current_sequence)
            predictions.append(next_pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
            
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    def predict_arima_garch(self, data: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ARIMA+GARCH predictions with volatility estimates"""
        if 'arima_garch' not in self.models:
            raise ValueError("ARIMA+GARCH model not found")
            
        # Get predictions from ARIMA
        arima_forecast = self.models['arima_garch']['arima'].forecast(horizon)
        
        # Get volatility forecasts from GARCH
        garch_forecast = self.models['arima_garch']['garch'].forecast(horizon=horizon)
        volatility = np.sqrt(garch_forecast.variance.values[-horizon:])
        
        return arima_forecast.values, volatility
    
    def predict_prophet(self, data: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Generate Prophet predictions"""
        if 'prophet' not in self.models:
            raise ValueError("Prophet model not found")
            
        # Create future dates
        future_dates = pd.date_range(
            start=data.index[-1] + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        future = pd.DataFrame({'ds': future_dates})
        
        # Make predictions
        forecast = self.models['prophet'].predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def generate_ensemble_predictions(
        self,
        data: pd.DataFrame,
        horizon: int,
        scaler: MinMaxScaler = None
    ) -> Dict:
        """Generate ensemble predictions using all models"""
        predictions = {}
        weights = self.get_weights(horizon)
        
        try:
            # Get LSTM predictions
            lstm_preds = self.predict_lstm(data, horizon, scaler)
            predictions['lstm'] = {
                'values': lstm_preds.flatten(),
                'weight': weights['lstm']
            }
        except Exception as e:
            logger.error(f"Error in LSTM predictions: {e}")
            predictions['lstm'] = None
            
        try:
            # Get ARIMA+GARCH predictions
            arima_preds, volatility = self.predict_arima_garch(data, horizon)
            predictions['arima_garch'] = {
                'values': arima_preds,
                'volatility': volatility,
                'weight': weights['arima_garch']
            }
        except Exception as e:
            logger.error(f"Error in ARIMA+GARCH predictions: {e}")
            predictions['arima_garch'] = None
            
        try:
            # Get Prophet predictions
            prophet_preds = self.predict_prophet(data, horizon)
            predictions['prophet'] = {
                'values': prophet_preds['yhat'].values,
                'lower': prophet_preds['yhat_lower'].values,
                'upper': prophet_preds['yhat_upper'].values,
                'weight': weights['prophet']
            }
        except Exception as e:
            logger.error(f"Error in Prophet predictions: {e}")
            predictions['prophet'] = None
            
        # Calculate weighted ensemble predictions
        ensemble_predictions = np.zeros(horizon)
        total_weight = 0
        
        for model_name, pred_dict in predictions.items():
            if pred_dict is not None:
                ensemble_predictions += pred_dict['values'] * pred_dict['weight']
                total_weight += pred_dict['weight']
                
        if total_weight > 0:
            ensemble_predictions /= total_weight
            
        predictions['ensemble'] = {
            'values': ensemble_predictions,
            'weights': weights
        }
        
        return predictions
    
    def calculate_confidence_intervals(
        self,
        predictions: Dict,
        confidence_level: float = 0.95
    ) -> Dict:
        """Calculate confidence intervals for ensemble predictions"""
        if not all(k in predictions for k in ['arima_garch', 'prophet']):
            raise ValueError("Required model predictions not found")
            
        z_score = 1.96  # 95% confidence level
        
        # Get volatility estimates
        arima_volatility = predictions['arima_garch']['volatility']
        prophet_range = (
            predictions['prophet']['upper'] - predictions['prophet']['lower']
        ) / (2 * z_score)
        
        # Combine volatility estimates
        combined_volatility = np.sqrt(
            (arima_volatility ** 2 + prophet_range ** 2) / 2
        )
        
        # Calculate confidence intervals
        ensemble_values = predictions['ensemble']['values']
        lower_bound = ensemble_values - z_score * combined_volatility
        upper_bound = ensemble_values + z_score * combined_volatility
        
        predictions['ensemble'].update({
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'volatility': combined_volatility
        })
        
        return predictions 