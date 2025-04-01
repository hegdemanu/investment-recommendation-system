import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from typing import Tuple


class LSTMModel:
    """
    LSTM model for time series prediction
    """
    
    def __init__(self, lookback_days: int = 60, prediction_days: int = 1):
        """
        Initialize the LSTM model
        
        Args:
            lookback_days: Number of past days to use for prediction
            prediction_days: Number of future days to predict
        """
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            data: DataFrame with stock prices (should have 'Close' column)
            
        Returns:
            X: Input data
            y: Target data
        """
        # Scale data
        scaled_data = self.scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        X = []
        y = []
        
        for i in range(self.lookback_days, len(scaled_data) - self.prediction_days):
            X.append(scaled_data[i - self.lookback_days:i, 0])
            y.append(scaled_data[i + self.prediction_days - 1, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        # Reshape data for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build the LSTM model
        
        Args:
            input_shape: Shape of the input data
        """
        self.model = Sequential()
        
        # LSTM layers
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(0.2))
        
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        
        # Output layer
        self.model.add(Dense(units=1))
        
        # Compile model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
    def train(self, data: pd.DataFrame, epochs: int = 25, batch_size: int = 32, validation_split: float = 0.2) -> dict:
        """
        Train the LSTM model
        
        Args:
            data: DataFrame with stock prices (should have 'Close' column)
            epochs: Number of epochs to train
            batch_size: Batch size
            validation_split: Validation split ratio
            
        Returns:
            History object
        """
        X, y = self._prepare_data(data)
        
        if self.model is None:
            self.build_model((X.shape[1], 1))
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history.history
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            data: DataFrame with stock prices (should have 'Close' column)
            
        Returns:
            Predicted prices
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train() first.")
        
        # Get the last lookback_days of data
        latest_data = data['Close'].values[-self.lookback_days:]
        latest_data_scaled = self.scaler.transform(latest_data.reshape(-1, 1))
        
        X_test = []
        X_test.append(latest_data_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Make prediction
        predicted_price_scaled = self.model.predict(X_test)
        predicted_price = self.scaler.inverse_transform(predicted_price_scaled)
        
        return predicted_price[0, 0]
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train() first.")
        
        self.model.save(filepath)
        
    def load_model(self, filepath: str) -> None:
        """
        Load the model from disk
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath) 