#!/usr/bin/env python3
"""
Model Trainer for Investment Recommendation System
Implements training pipelines for LSTM, ARIMA+GARCH, and Prophet models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from pathlib import Path
import time

import tensorflow as tf
from prophet import Prophet
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Local imports
from ..utils.model_registry import ModelRegistry
from ..utils.data_processor import DataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_registry: ModelRegistry = None, config: Dict = None):
        """Initialize model trainer with registry"""
        self.model_registry = model_registry or ModelRegistry()
        self.data_processor = DataProcessor()
        self.config = config or self._default_config()
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
    
    def _fetch_training_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch stock data for training"""
        logger.info(f"Fetching training data for {symbol}")
        data = self.data_processor.get_stock_data(symbol, period=period, interval="1d")
        
        # Add technical indicators
        data = self.data_processor.add_technical_indicators(data)
        
        return data
    
    def _prepare_features(self, data: pd.DataFrame, target_col: str = 'Close') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variable"""
        features = data.copy()
        target = features[target_col].copy()
        
        # Drop non-feature columns
        features.drop(columns=['Volume', 'Open', 'High', 'Low'], inplace=True, errors='ignore')
        
        # Handle missing values
        features.fillna(method='ffill', inplace=True)
        features.fillna(0, inplace=True)
        
        return features, target
    
    def _prepare_sequences(
        self, 
        data: pd.DataFrame, 
        target_col: str = 'Close', 
        seq_length: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data sequences for LSTM"""
        seq_length = seq_length or self.config['lstm']['sequence_length']
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[[target_col]])
        self.scalers[target_col] = scaler
        
        X, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X.append(scaled_data[i-seq_length:i])
            y.append(scaled_data[i])
        
        return np.array(X), np.array(y), scaler
    
    def _build_lstm_model(self, input_shape: Tuple) -> tf.keras.Model:
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
    
    def train_lstm(self, symbol: str, target_col: str = 'Close') -> Dict:
        """Train LSTM model for a symbol"""
        logger.info(f"Training LSTM model for {symbol}")
        start_time = time.time()
        
        # Fetch data
        data = self._fetch_training_data(symbol)
        
        # Prepare sequences
        X, y, scaler = self._prepare_sequences(data, target_col)
        
        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build and train model
        model = self._build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
        
        history = model.fit(
            X_train, y_train,
            epochs=self.config['lstm']['epochs'],
            batch_size=self.config['lstm']['batch_size'],
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate model
        y_pred = model.predict(X_test, verbose=0)
        
        # Transform predictions and actual values back to original scale
        y_test_inv = scaler.inverse_transform(y_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        # Calculate accuracy as 1 - normalized RMSE
        price_range = data[target_col].max() - data[target_col].min()
        accuracy = max(0, 1 - (rmse / price_range))
        
        training_time = time.time() - start_time
        
        # Register model with registry
        model_id = self.model_registry.register_model(
            model_type="LSTM",
            model_object=model,
            symbol=symbol,
            metrics={
                "accuracy": accuracy,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "trainingTime": training_time
            },
            features=data.columns.tolist(),
            hyperparameters=self.config['lstm']
        )
        
        logger.info(f"LSTM model for {symbol} trained and registered as {model_id}")
        
        return {
            "model_id": model_id,
            "model_type": "LSTM",
            "metrics": {
                "accuracy": accuracy,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            },
            "training_time": training_time,
            "history": history.history
        }
    
    def train_arima_garch(self, symbol: str, target_col: str = 'Close') -> Dict:
        """Train ARIMA+GARCH model for a symbol"""
        logger.info(f"Training ARIMA+GARCH model for {symbol}")
        start_time = time.time()
        
        # Fetch data
        data = self._fetch_training_data(symbol)
        
        # Split data
        split = int(len(data) * 0.8)
        train_data = data.iloc[:split]
        test_data = data.iloc[split:]
        
        # Train ARIMA
        arima_model = ARIMA(
            train_data[target_col],
            order=self.config['arima']['order'],
            seasonal_order=self.config['arima']['seasonal_order']
        )
        arima_results = arima_model.fit()
        
        # Train GARCH on residuals
        residuals = arima_results.resid
        garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
        garch_results = garch_model.fit(disp='off')
        
        # Make forecasts for test period
        forecasts = arima_results.forecast(steps=len(test_data))
        
        # Calculate metrics
        mse = mean_squared_error(test_data[target_col], forecasts)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data[target_col], forecasts)
        
        # Calculate accuracy as 1 - normalized RMSE
        price_range = data[target_col].max() - data[target_col].min()
        accuracy = max(0, 1 - (rmse / price_range))
        
        training_time = time.time() - start_time
        
        # Combine models
        combined_model = {
            'arima': arima_results,
            'garch': garch_results
        }
        
        # Register model with registry
        model_id = self.model_registry.register_model(
            model_type="ARIMA",
            model_object=combined_model,
            symbol=symbol,
            metrics={
                "accuracy": accuracy,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "trainingTime": training_time
            },
            hyperparameters={
                "arima_order": self.config['arima']['order'],
                "seasonal_order": self.config['arima']['seasonal_order'],
                "garch_p": 1,
                "garch_q": 1
            }
        )
        
        logger.info(f"ARIMA+GARCH model for {symbol} trained and registered as {model_id}")
        
        return {
            "model_id": model_id,
            "model_type": "ARIMA",
            "metrics": {
                "accuracy": accuracy,
                "mse": mse,
                "rmse": rmse,
                "mae": mae
            },
            "training_time": training_time
        }
    
    def train_prophet(self, symbol: str, target_col: str = 'Close') -> Dict:
        """Train Prophet model for a symbol"""
        logger.info(f"Training Prophet model for {symbol}")
        start_time = time.time()
        
        # Fetch data
        data = self._fetch_training_data(symbol)
        
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data[target_col]
        })
        
        # Split data
        split = int(len(prophet_data) * 0.8)
        train_data = prophet_data.iloc[:split]
        test_data = prophet_data.iloc[split:]
        
        # Initialize and train Prophet model
        prophet_model = Prophet(
            changepoint_prior_scale=self.config['prophet']['changepoint_prior_scale'],
            seasonality_prior_scale=self.config['prophet']['seasonality_prior_scale'],
            seasonality_mode=self.config['prophet']['seasonality_mode']
        )
        prophet_model.fit(train_data)
        
        # Make predictions for test period
        future = prophet_model.make_future_dataframe(periods=len(test_data))
        forecast = prophet_model.predict(future)
        
        # Extract predictions for test period
        predictions = forecast.iloc[-len(test_data):]['yhat'].values
        
        # Calculate metrics
        mse = mean_squared_error(test_data['y'], predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data['y'], predictions)
        
        # Calculate accuracy as 1 - normalized RMSE
        price_range = data[target_col].max() - data[target_col].min()
        accuracy = max(0, 1 - (rmse / price_range))
        
        training_time = time.time() - start_time
        
        # Register model with registry
        model_id = self.model_registry.register_model(
            model_type="PROPHET",
            model_object=prophet_model,
            symbol=symbol,
            metrics={
                "accuracy": accuracy,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "trainingTime": training_time
            },
            hyperparameters=self.config['prophet']
        )
        
        logger.info(f"Prophet model for {symbol} trained and registered as {model_id}")
        
        return {
            "model_id": model_id,
            "model_type": "PROPHET",
            "metrics": {
                "accuracy": accuracy,
                "mse": mse,
                "rmse": rmse,
                "mae": mae
            },
            "training_time": training_time
        }
    
    def train(self, symbol: str, model_type: str = "LSTM", options: Dict = None) -> Dict:
        """Train a specific model type for a symbol"""
        if options:
            # Update config with provided options
            if model_type.upper() == "LSTM" and 'lstm' in options:
                self.config['lstm'].update(options['lstm'])
            elif model_type.upper() == "ARIMA" and 'arima' in options:
                self.config['arima'].update(options['arima'])
            elif model_type.upper() == "PROPHET" and 'prophet' in options:
                self.config['prophet'].update(options['prophet'])
        
        if model_type.upper() == "LSTM":
            return self.train_lstm(symbol)
        elif model_type.upper() == "ARIMA":
            return self.train_arima_garch(symbol)
        elif model_type.upper() == "PROPHET":
            return self.train_prophet(symbol)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train_all_models(self, symbol: str) -> Dict[str, Dict]:
        """Train all model types for a symbol"""
        results = {}
        
        try:
            results["LSTM"] = self.train_lstm(symbol)
        except Exception as e:
            logger.error(f"Error training LSTM for {symbol}: {e}")
            results["LSTM"] = {"error": str(e)}
            
        try:
            results["ARIMA"] = self.train_arima_garch(symbol)
        except Exception as e:
            logger.error(f"Error training ARIMA+GARCH for {symbol}: {e}")
            results["ARIMA"] = {"error": str(e)}
            
        try:
            results["PROPHET"] = self.train_prophet(symbol)
        except Exception as e:
            logger.error(f"Error training Prophet for {symbol}: {e}")
            results["PROPHET"] = {"error": str(e)}
            
        return results 