"""
Module for training and managing prediction models.
Supports LSTM, ARIMA-GARCH, and Prophet models.
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

class ModelTrainer:
    """
    Module for training and managing prediction models.
    Supports LSTM, ARIMA-GARCH, and Prophet models.
    """
    
    def __init__(self, models_dir="./models"):
        """
        Initialize the ModelTrainer module.
        
        Parameters:
        -----------
        models_dir : str, optional
            Directory to save trained models
        """
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"Created models directory: {models_dir}")
    
    def check_data_drift(self, new_data, threshold=0.1):
        """
        Check if there's significant data drift requiring model retraining.
        
        Parameters:
        -----------
        new_data : pd.DataFrame
            New data to check for drift
        threshold : float, optional
            Threshold for determining significant drift
            
        Returns:
        --------
        bool : Whether significant drift is detected
        """
        stats_path = os.path.join(self.models_dir, "data_stats.csv")
        
        # If no previous stats exist, save current and return True
        if not os.path.exists(stats_path):
            self._save_data_stats(new_data, stats_path)
            print("No previous data statistics found. Saving current stats.")
            return True
        
        # Load previous statistics
        prev_stats = pd.read_csv(stats_path)
        
        # Calculate statistics for new data
        new_stats = self._calculate_data_stats(new_data)
        
        # Calculate drift for each feature
        drift_scores = []
        for feature in new_stats.columns:
            if feature in prev_stats.columns:
                prev_mean = prev_stats[feature].iloc[0]
                new_mean = new_stats[feature].iloc[0]
                
                if prev_mean != 0:
                    drift = abs(new_mean - prev_mean) / abs(prev_mean)
                    drift_scores.append(drift)
                    print(f"Drift for {feature}: {drift:.4f}")
        
        # Calculate average drift
        if drift_scores:
            avg_drift = np.mean(drift_scores)
            print(f"Average drift across features: {avg_drift:.4f}")
            
            # Update statistics if drift is significant
            if avg_drift > threshold:
                print(f"Significant data drift detected: {avg_drift:.4f} > {threshold}. Retraining recommended.")
                self._save_data_stats(new_data, stats_path)
                return True
            else:
                print(f"No significant data drift detected: {avg_drift:.4f} <= {threshold}")
                return False
        else:
            print("Could not calculate drift scores. Retraining recommended.")
            return True
    
    def _calculate_data_stats(self, data):
        """
        Calculate statistics for data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to calculate statistics for
            
        Returns:
        --------
        pd.DataFrame : Calculated statistics
        """
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Calculate statistics
        stats = pd.DataFrame({
            col: [numeric_data[col].mean()] for col in numeric_data.columns
        })
        
        return stats
    
    def _save_data_stats(self, data, stats_path):
        """
        Save data statistics to file.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to calculate statistics for
        stats_path : str
            Path to save statistics file
        """
        stats = self._calculate_data_stats(data)
        stats.to_csv(stats_path, index=False)
        print(f"Data statistics saved to {stats_path}")
    
    def train_lstm_model(self, data, ticker, seq_length=60, epochs=100, batch_size=32):
        """
        Train an LSTM model for a specific ticker.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data for model training
        ticker : str
            The ticker symbol to train model for
        seq_length : int, optional
            Sequence length for LSTM input
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size for training
            
        Returns:
        --------
        tuple : (trained model, scaler, features used)
        """
        print(f"Training LSTM model for {ticker}...")
        
        # Filter data for the specific ticker
        ticker_data = data[data['ticker'] == ticker].sort_values('Date')
        
        # Select features for LSTM
        base_features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
        available_features = [f for f in base_features if f in ticker_data.columns]
        
        if len(available_features) < 3:
            print(f"Not enough features for {ticker}, need at least 3 features.")
            return None, None, None
        
        print(f"Using features: {available_features}")
        
        # Extract features
        X = ticker_data[available_features].values
        
        # Scale the features
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        
        # Create sequences for LSTM
        X_seq, y_seq = [], []
        
        for i in range(len(X_scaled) - seq_length):
            X_seq.append(X_scaled[i:i+seq_length])
            # Predict the next day's price (index 0 corresponds to 'Price')
            y_seq.append(X_scaled[i+seq_length, 0])
        
        if not X_seq:
            print(f"Not enough data for {ticker} to create sequences. Need at least {seq_length+1} data points.")
            return None, None, None
        
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        
        # Split into train and validation sets
        train_size = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:train_size], X_seq[train_size:]
        y_train, y_val = y_seq[:train_size], y_seq[train_size:]
        
        print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model with early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Training and Validation Loss for {ticker}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.models_dir, f"{ticker}_training_loss.png"))
        plt.close()
        
        # Save model
        model_path = os.path.join(self.models_dir, f"{ticker}_lstm.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, f"{ticker}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {scaler_path}")
        
        # Save features used
        features_path = os.path.join(self.models_dir, f"{ticker}_features.pkl")
        with open(features_path, 'wb') as f:
            pickle.dump(available_features, f)
        print(f"Features list saved to {features_path}")
        
        return model, scaler, available_features
    
    def train_lstm_models(self, data):
        """
        Train LSTM models for all tickers in the data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data for model training
            
        Returns:
        --------
        dict : Dictionary of trained models, scalers, and features
        """
        result = {
            "models": {},
            "scalers": {},
            "features": {}
        }
        
        unique_tickers = data['ticker'].unique()
        print(f"Training LSTM models for {len(unique_tickers)} tickers...")
        
        for ticker in unique_tickers:
            model, scaler, features = self.train_lstm_model(data, ticker)
            
            if model is not None:
                result["models"][ticker] = model
                result["scalers"][ticker] = scaler
                result["features"][ticker] = features
        
        print(f"Successfully trained models for {len(result['models'])} tickers.")
        return result
    
    def load_lstm_models(self):
        """
        Load pre-trained LSTM models.
        
        Returns:
        --------
        dict : Dictionary of loaded models, scalers, and features
        """
        result = {
            "models": {},
            "scalers": {},
            "features": {}
        }
        
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith("_lstm.h5")]
        
        if not model_files:
            print("No trained models found.")
            return result
        
        print(f"Loading {len(model_files)} pre-trained LSTM models...")
        
        for model_file in model_files:
            ticker = model_file.replace("_lstm.h5", "")
            model_path = os.path.join(self.models_dir, model_file)
            scaler_path = os.path.join(self.models_dir, f"{ticker}_scaler.pkl")
            features_path = os.path.join(self.models_dir, f"{ticker}_features.pkl")
            
            try:
                # Load model
                model = load_model(model_path)
                
                # Load scaler
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                
                # Load features
                with open(features_path, 'rb') as f:
                    features = pickle.load(f)
                
                result["models"][ticker] = model
                result["scalers"][ticker] = scaler
                result["features"][ticker] = features
                
                print(f"Successfully loaded model for {ticker}")
            except Exception as e:
                print(f"Error loading model for {ticker}: {str(e)}")
        
        print(f"Successfully loaded {len(result['models'])} models.")
        return result
    
    def predict_lstm(self, models_dict, data, horizon="short"):
        """
        Generate predictions using trained LSTM models.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of trained models, scalers, and features
        data : pd.DataFrame
            Data to base predictions on
        horizon : str, optional
            Time horizon for predictions - "short", "mid", or "long"
            
        Returns:
        --------
        pd.DataFrame : Predictions
        """
        predictions = []
        
        # Define the number of steps to predict based on horizon
        if horizon == "short":
            prediction_days = [1, 5, 21]  # 1 day, 1 week, 1 month
            labels = ["next_day", "next_week", "next_month"]
        elif horizon == "mid":
            prediction_days = [63, 126]  # ~3 months, ~6 months
            labels = ["next_quarter", "next_half_year"]
        elif horizon == "long":
            prediction_days = [252]  # ~1 year
            labels = ["next_year"]
        else:
            raise ValueError("Invalid horizon. Choose 'short', 'mid', or 'long'.")
        
        seq_length = 60  # Same as used in training
        
        for ticker, ticker_data in data.groupby('ticker'):
            if ticker not in models_dict["models"]:
                print(f"No model found for {ticker}, skipping predictions...")
                continue
            
            model = models_dict["models"][ticker]
            scaler = models_dict["scalers"][ticker]
            features = models_dict["features"][ticker]
            
            # Sort by date
            ticker_data = ticker_data.sort_values('Date')
            
            # Select available features
            available_features = [f for f in features if f in ticker_data.columns]
            X = ticker_data[available_features].values
            
            # Scale the features
            X_scaled = scaler.transform(X)
            
            # Create input sequence for prediction
            if len(X_scaled) < seq_length:
                print(f"Not enough data for {ticker} (need at least {seq_length} data points)")
                continue
            
            # Get latest date and price
            latest_date = ticker_data.iloc[-1]['Date']
            latest_price = ticker_data.iloc[-1]['Price'] if 'Price' in ticker_data.columns else ticker_data.iloc[-1]['Close']
            
            ticker_pred = {
                'ticker': ticker,
                'last_date': latest_date,
                'latest_price': latest_price
            }
            
            # Make predictions for each time horizon
            for days, label in zip(prediction_days, labels):
                # Start with the last sequence of data
                pred_sequence = X_scaled[-seq_length:].copy()
                
                # Iteratively predict future days
                for day in range(days):
                    # Reshape for prediction
                    X_pred = pred_sequence[-seq_length:].reshape(1, seq_length, len(available_features))
                    
                    # Predict next value (scaled)
                    next_scaled_price = model.predict(X_pred)[0][0]
                    
                    # Create a new row for the prediction
                    new_row = pred_sequence[-1:].copy()
                    new_row[0, 0] = next_scaled_price  # Update price
                    
                    # Add prediction to sequence for next iteration
                    pred_sequence = np.vstack([pred_sequence, new_row])
                
                # Get final prediction and convert back to original scale
                final_scaled = pred_sequence[-1:].copy()
                final_scaled_array = np.zeros_like(final_scaled)
                final_scaled_array[0, 0] = final_scaled[0, 0]
                
                # Inverse transform to get the actual price prediction
                predicted_price = scaler.inverse_transform(final_scaled_array)[0, 0]
                
                # Calculate percentage change
                predicted_change = ((predicted_price / latest_price) - 1) * 100
                
                # Add to the prediction dictionary
                ticker_pred[f'{label}_price'] = predicted_price
                ticker_pred[f'{label}_change'] = predicted_change
            
            predictions.append(ticker_pred)
        
        if predictions:
            return pd.DataFrame(predictions)
        else:
            print("No predictions generated.")
            return pd.DataFrame() 