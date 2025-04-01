#!/usr/bin/env python3
"""
Training Pipeline for Investment Recommendation System
Implements LSTM, ARIMA+GARCH, and Prophet models with sentiment integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
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

class ModelTrainingPipeline:
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        
    def _default_config(self) -> Dict:
        return {
            'lstm': {
                'units': 50,
                'dropout': 0.2,
                'epochs': 100,
                'batch_size': 32,
                'sequence_length': 60
            },
            'arima': {
                'order': (5,1,2),
                'seasonal_order': (1,1,1,12)
            },
            'prophet': {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10,
                'seasonality_mode': 'multiplicative'
            }
        }
    
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[[target_col]])
        self.scalers[target_col] = scaler
        
        X, y = [], []
        for i in range(self.config['lstm']['sequence_length'], len(scaled_data)):
            X.append(scaled_data[i-self.config['lstm']['sequence_length']:i])
            y.append(scaled_data[i])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple) -> tf.keras.Model:
        """Build LSTM model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                units=self.config['lstm']['units'],
                return_sequences=True,
                input_shape=input_shape
            ),
            tf.keras.layers.Dropout(self.config['lstm']['dropout']),
            tf.keras.layers.LSTM(units=self.config['lstm']['units']),
            tf.keras.layers.Dropout(self.config['lstm']['dropout']),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_lstm(self, data: pd.DataFrame, target_col: str = 'Close') -> Dict:
        """Train LSTM model"""
        logger.info("Training LSTM model...")
        X, y = self.prepare_data(data, target_col)
        
        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build and train model
        model = self.build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
        history = model.fit(
            X_train, y_train,
            epochs=self.config['lstm']['epochs'],
            batch_size=self.config['lstm']['batch_size'],
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        self.models['lstm'] = model
        return {
            'model': model,
            'history': history.history,
            'scaler': self.scalers[target_col]
        }
    
    def train_arima_garch(self, data: pd.DataFrame, target_col: str = 'Close') -> Dict:
        """Train ARIMA+GARCH model"""
        logger.info("Training ARIMA+GARCH model...")
        
        # Train ARIMA
        arima_model = ARIMA(
            data[target_col],
            order=self.config['arima']['order'],
            seasonal_order=self.config['arima']['seasonal_order']
        )
        arima_results = arima_model.fit()
        
        # Train GARCH on residuals
        residuals = arima_results.resid
        garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
        garch_results = garch_model.fit(disp='off')
        
        self.models['arima_garch'] = {
            'arima': arima_results,
            'garch': garch_results
        }
        
        return {
            'arima_results': arima_results,
            'garch_results': garch_results
        }
    
    def train_prophet(self, data: pd.DataFrame, target_col: str = 'Close') -> Dict:
        """Train Prophet model"""
        logger.info("Training Prophet model...")
        
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data[target_col]
        })
        
        # Initialize and train Prophet model
        prophet_model = Prophet(
            changepoint_prior_scale=self.config['prophet']['changepoint_prior_scale'],
            seasonality_prior_scale=self.config['prophet']['seasonality_prior_scale'],
            seasonality_mode=self.config['prophet']['seasonality_mode']
        )
        prophet_model.fit(prophet_data)
        
        self.models['prophet'] = prophet_model
        return {'model': prophet_model}
    
    def train_all_models(self, data: pd.DataFrame, target_col: str = 'Close') -> Dict:
        """Train all models and return results"""
        results = {}
        
        try:
            results['lstm'] = self.train_lstm(data, target_col)
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            results['lstm'] = None
            
        try:
            results['arima_garch'] = self.train_arima_garch(data, target_col)
        except Exception as e:
            logger.error(f"Error training ARIMA+GARCH: {e}")
            results['arima_garch'] = None
            
        try:
            results['prophet'] = self.train_prophet(data, target_col)
        except Exception as e:
            logger.error(f"Error training Prophet: {e}")
            results['prophet'] = None
            
        return results
    
    def save_models(self, save_dir: str = 'models/'):
        """Save trained models"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save LSTM model
        if 'lstm' in self.models:
            self.models['lstm'].save(save_path / 'lstm_model')
            
        # Save ARIMA+GARCH parameters
        if 'arima_garch' in self.models:
            pd.to_pickle(self.models['arima_garch'], save_path / 'arima_garch_model.pkl')
            
        # Save Prophet model
        if 'prophet' in self.models:
            pd.to_pickle(self.models['prophet'], save_path / 'prophet_model.pkl')
            
        logger.info(f"Models saved to {save_path}") 