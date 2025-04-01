import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import json
import os
from typing import Tuple, Dict, Any
import yfinance as yf
import ta

class StockPredictor:
    def __init__(self, model_registry):
        self.model_registry = model_registry
        self.lookback_period = 60  # Days of historical data to use
        
    def _load_model_components(self, symbol: str) -> Tuple[Any, Any, Dict]:
        """Load model, scaler and metadata for a given symbol"""
        try:
            # Get model paths from registry
            model_path = self.model_registry.get_model_path("lstm", symbol, "model")
            scaler_path = self.model_registry.get_model_path("lstm", symbol, "scaler") 
            metadata = self.model_registry.get_model_metadata("lstm", symbol)
            
            if not all([model_path, scaler_path, metadata]):
                raise ValueError(f"Missing model components for {symbol}")
            
            # Load components
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            
            return model, scaler, metadata
            
        except Exception as e:
            raise Exception(f"Error loading model components: {str(e)}")
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare feature set from raw data"""
        df = data.copy()
        
        # Technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        df['bb_high'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
        df['bb_low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
        df['volume_sma'] = ta.volume.SMAIndicator(df['Volume'], window=20).sma_indicator()
        
        # Price changes
        df['price_change'] = df['Close'].pct_change()
        df['volume_change'] = df['Volume'].pct_change()
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        # Select features
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'rsi', 'macd', 'bb_high', 'bb_low', 'volume_sma',
                   'price_change', 'volume_change']
                   
        return df[features].values
    
    def _fetch_historical_data(self, symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
        """Fetch historical data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
                
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")
    
    def predict(self, symbol: str, features: list = None) -> Tuple[float, float]:
        """
        Make prediction for a symbol
        Returns: (prediction, confidence)
        """
        try:
            # Load model components
            model, scaler, metadata = self._load_model_components(symbol)
            
            # If features not provided, fetch latest data
            if not features:
                data = self._fetch_historical_data(symbol)
                features = self._prepare_features(data)
            
            # Scale features
            scaled_features = scaler.transform(features)
            
            # Prepare sequence
            sequence = scaled_features[-self.lookback_period:]
            sequence = sequence.reshape((1, self.lookback_period, scaled_features.shape[1]))
            
            # Make prediction
            prediction = model.predict(sequence)
            
            # Inverse transform prediction
            prediction = scaler.inverse_transform(prediction.reshape(1, -1))
            prediction = prediction[0][0]  # Get scalar value
            
            # Calculate confidence based on historical accuracy
            current_price = features[-1][3]  # Close price
            confidence = 1.0 - abs(prediction - current_price) / current_price
            confidence = max(0.0, min(1.0, confidence))  # Clip between 0 and 1
            
            return prediction, confidence
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
            
    def get_model_info(self, symbol: str) -> Dict:
        """Get model information for a symbol"""
        try:
            return self.model_registry.get_model_metadata("lstm", symbol)
        except Exception as e:
            raise Exception(f"Error getting model info: {str(e)}") 