import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
import os
from datetime import datetime
from typing import Dict, Tuple, List

class ModelTrainer:
    def __init__(self, model_registry):
        self.model_registry = model_registry
        self.lookback_period = 60
        self.train_split = 0.8
        self.batch_size = 32
        self.epochs = 50
        
    def _fetch_training_data(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        """Fetch historical data for training"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
                
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching training data: {str(e)}")
    
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
    
    def _prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        
        for i in range(len(data) - self.lookback_period):
            X.append(data[i:(i + self.lookback_period)])
            y.append(data[i + self.lookback_period, 3])  # Next day's close price
            
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: tuple) -> tf.keras.Model:
        """Build LSTM model"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    async def train(self, symbol: str) -> bool:
        """Train a new model for a symbol"""
        try:
            # Fetch and prepare data
            data = self._fetch_training_data(symbol)
            features = self._prepare_features(data)
            
            # Scale features
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Prepare sequences
            X, y = self._prepare_sequences(scaled_features)
            
            # Split into train/test
            split_idx = int(len(X) * self.train_split)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Build and train model
            model = self._build_model((self.lookback_period, X.shape[2]))
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0
            )
            
            # Evaluate model
            test_loss = model.evaluate(X_test, y_test, verbose=0)
            
            # Save model components
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metadata = {
                "symbol": symbol,
                "type": "lstm",
                "created_at": timestamp,
                "metrics": {
                    "test_loss": float(test_loss[0]),
                    "test_mae": float(test_loss[1])
                },
                "parameters": {
                    "lookback_period": self.lookback_period,
                    "train_split": self.train_split,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs
                },
                "features": [
                    "Open", "High", "Low", "Close", "Volume",
                    "rsi", "macd", "bb_high", "bb_low", "volume_sma",
                    "price_change", "volume_change"
                ]
            }
            
            # Register model with registry
            self.model_registry.register_model(
                model_type="lstm",
                symbol=symbol,
                model=model,
                scaler=scaler,
                metadata=metadata
            )
            
            return True
            
        except Exception as e:
            raise Exception(f"Training error: {str(e)}")
    
    def get_training_history(self, symbol: str) -> List[Dict]:
        """Get training history for a symbol"""
        try:
            return self.model_registry.get_model_history("lstm", symbol)
        except Exception as e:
            raise Exception(f"Error getting training history: {str(e)}")
            
    def evaluate_model(self, symbol: str) -> Dict:
        """Evaluate model performance on recent data"""
        try:
            # Load model components
            model_path = self.model_registry.get_model_path("lstm", symbol, "model")
            scaler_path = self.model_registry.get_model_path("lstm", symbol, "scaler")
            
            if not all([model_path, scaler_path]):
                raise ValueError(f"No model found for {symbol}")
            
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            
            # Get recent data
            data = self._fetch_training_data(symbol, period="6mo")
            features = self._prepare_features(data)
            scaled_features = scaler.transform(features)
            
            # Prepare sequences
            X, y = self._prepare_sequences(scaled_features)
            
            # Evaluate
            metrics = model.evaluate(X, y, verbose=0)
            predictions = model.predict(X)
            
            # Calculate additional metrics
            mse = np.mean((predictions - y) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - y))
            
            return {
                "loss": float(metrics[0]),
                "mae": float(metrics[1]),
                "mse": float(mse),
                "rmse": float(rmse),
                "samples": len(y)
            }
            
        except Exception as e:
            raise Exception(f"Evaluation error: {str(e)}") 