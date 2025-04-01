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

# Local imports
from ..utils.model_registry import ModelRegistry
from ..utils.data_processor import DataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self, model_registry: ModelRegistry = None, config: Dict = None):
        """Initialize prediction pipeline with model registry"""
        self.model_registry = model_registry or ModelRegistry()
        self.data_processor = DataProcessor()
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
        if horizon <= 5:
            return self.config['short_term']
        elif horizon <= 15:
            return self.config['medium_term']
        else:
            return self.config['long_term']
    
    def load_models(self, symbol: str) -> Dict:
        """Load all available models for a symbol"""
        models = {}
        
        # Load LSTM model
        lstm_model_data = self.model_registry.get_latest_model(symbol, "LSTM")
        if lstm_model_data:
            models['lstm'] = lstm_model_data['model']
        
        # Load ARIMA+GARCH model
        arima_model_data = self.model_registry.get_latest_model(symbol, "ARIMA")
        if arima_model_data:
            models['arima_garch'] = arima_model_data['model']
        
        # Load Prophet model
        prophet_model_data = self.model_registry.get_latest_model(symbol, "PROPHET")
        if prophet_model_data:
            models['prophet'] = prophet_model_data['model']
        
        return models
    
    def predict_lstm(self, data: pd.DataFrame, horizon: int, model, scaler: MinMaxScaler) -> np.ndarray:
        """Generate LSTM predictions"""
        if model is None:
            raise ValueError("LSTM model not found")
            
        # Prepare sequence
        sequence = scaler.transform(data[['Close']].tail(60))
        sequence = sequence.reshape((1, 60, 1))
        
        # Generate predictions
        predictions = []
        current_sequence = sequence
        
        for _ in range(horizon):
            # Predict next value
            next_pred = model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
            
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    def predict_arima_garch(self, data: pd.DataFrame, horizon: int, model) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ARIMA+GARCH predictions with volatility estimates"""
        if model is None:
            raise ValueError("ARIMA+GARCH model not found")
            
        # Get predictions from ARIMA
        arima_forecast = model['arima'].forecast(horizon)
        
        # Get volatility forecasts from GARCH
        garch_forecast = model['garch'].forecast(horizon=horizon)
        volatility = np.sqrt(garch_forecast.variance.values[-horizon:])
        
        return arima_forecast.values, volatility
    
    def predict_prophet(self, data: pd.DataFrame, horizon: int, model) -> pd.DataFrame:
        """Generate Prophet predictions"""
        if model is None:
            raise ValueError("Prophet model not found")
            
        # Create future dates
        future_dates = pd.date_range(
            start=data.index[-1] + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        future = pd.DataFrame({'ds': future_dates})
        
        # Make predictions
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def generate_predictions(
        self,
        symbol: str,
        horizon: int = 5,
        model_type: str = None
    ) -> Dict:
        """Generate predictions for a symbol using specified or ensemble models"""
        # Get stock data
        data = self.data_processor.get_stock_data(symbol, period="1y", interval="1d")
        
        # Load models
        models = self.load_models(symbol)
        
        # If specific model type requested
        if model_type and model_type != "ENSEMBLE":
            model_key = model_type.lower()
            if model_key not in models:
                raise ValueError(f"Model {model_type} not found for {symbol}")
            
            if model_key == "lstm":
                # Get scaler from model registry or create new one
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data[['Close']])
                
                predictions = self.predict_lstm(data, horizon, models['lstm'], scaler)
                
                # Create prediction dates
                last_date = data.index[-1]
                prediction_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
                
                # Create confidence intervals (simple estimate)
                volatility = data['Close'].std() * 0.01 * np.sqrt(np.arange(1, horizon + 1))
                upper_bound = predictions.flatten() + 1.96 * volatility
                lower_bound = predictions.flatten() - 1.96 * volatility
                
                return {
                    "symbol": symbol,
                    "modelType": "LSTM",
                    "predictions": [
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "price": float(price),
                            "upperBound": float(upper),
                            "lowerBound": float(lower)
                        }
                        for date, price, upper, lower in zip(
                            prediction_dates, predictions.flatten(),
                            upper_bound, lower_bound
                        )
                    ],
                    "confidence": 0.9
                }
            
            elif model_key == "arima_garch":
                predictions, volatility = self.predict_arima_garch(data, horizon, models['arima_garch'])
                
                # Create prediction dates
                last_date = data.index[-1]
                prediction_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
                
                # Create confidence intervals
                upper_bound = predictions + 1.96 * volatility
                lower_bound = predictions - 1.96 * volatility
                
                return {
                    "symbol": symbol,
                    "modelType": "ARIMA",
                    "predictions": [
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "price": float(price),
                            "upperBound": float(upper),
                            "lowerBound": float(lower)
                        }
                        for date, price, upper, lower in zip(
                            prediction_dates, predictions,
                            upper_bound, lower_bound
                        )
                    ],
                    "confidence": 0.85
                }
            
            elif model_key == "prophet":
                forecast = self.predict_prophet(data, horizon, models['prophet'])
                
                return {
                    "symbol": symbol,
                    "modelType": "PROPHET",
                    "predictions": [
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "price": float(yhat),
                            "upperBound": float(upper),
                            "lowerBound": float(lower)
                        }
                        for date, yhat, upper, lower in zip(
                            forecast['ds'],
                            forecast['yhat'],
                            forecast['yhat_upper'],
                            forecast['yhat_lower']
                        )
                    ],
                    "confidence": 0.8
                }
        
        # Generate ensemble predictions
        predictions = self.generate_ensemble_predictions(data, models, horizon)
        
        # Create prediction dates
        last_date = data.index[-1]
        prediction_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
        
        return {
            "symbol": symbol,
            "modelType": "ENSEMBLE",
            "predictions": [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "price": float(price),
                    "upperBound": float(upper),
                    "lowerBound": float(lower)
                }
                for date, price, upper, lower in zip(
                    prediction_dates,
                    predictions['ensemble']['values'],
                    predictions['ensemble'].get('upper_bound', [None] * horizon),
                    predictions['ensemble'].get('lower_bound', [None] * horizon)
                )
            ],
            "confidence": 0.95
        }
    
    def generate_ensemble_predictions(
        self,
        data: pd.DataFrame,
        models: Dict,
        horizon: int
    ) -> Dict:
        """Generate ensemble predictions using all available models"""
        predictions = {}
        weights = self.get_weights(horizon)
        
        # Prepare scaler for LSTM
        scaler = MinMaxScaler()
        scaler.fit_transform(data[['Close']])
        
        try:
            # Get LSTM predictions
            if 'lstm' in models:
                lstm_preds = self.predict_lstm(data, horizon, models['lstm'], scaler)
                predictions['lstm'] = {
                    'values': lstm_preds.flatten(),
                    'weight': weights['lstm']
                }
        except Exception as e:
            logger.error(f"Error in LSTM predictions: {e}")
            predictions['lstm'] = None
            
        try:
            # Get ARIMA+GARCH predictions
            if 'arima_garch' in models:
                arima_preds, volatility = self.predict_arima_garch(data, horizon, models['arima_garch'])
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
            if 'prophet' in models:
                prophet_preds = self.predict_prophet(data, horizon, models['prophet'])
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
        
        # Add confidence intervals if possible
        if 'arima_garch' in predictions and predictions['arima_garch'] is not None and \
           'prophet' in predictions and predictions['prophet'] is not None:
            predictions = self.calculate_confidence_intervals(predictions)
        
        return predictions
    
    def calculate_confidence_intervals(
        self,
        predictions: Dict,
        confidence_level: float = 0.95
    ) -> Dict:
        """Calculate confidence intervals for ensemble predictions"""
        if not all(k in predictions for k in ['arima_garch', 'prophet']):
            return predictions
            
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