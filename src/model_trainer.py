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
import logging
import json
from datetime import datetime

# Get logger
logger = logging.getLogger('ModelTrainer')

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
            logger.info(f"Created models directory: {models_dir}")
    
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
            logger.info("No previous data statistics found. Saving current stats.")
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
                    logger.debug(f"Drift for {feature}: {drift:.4f}")
        
        # Calculate average drift
        if drift_scores:
            avg_drift = np.mean(drift_scores)
            logger.info(f"Average drift across features: {avg_drift:.4f}")
            
            # Update statistics if drift is significant
            if avg_drift > threshold:
                logger.info(f"Significant data drift detected: {avg_drift:.4f} > {threshold}. Retraining recommended.")
                self._save_data_stats(new_data, stats_path)
                return True
            else:
                logger.info(f"No significant data drift detected: {avg_drift:.4f} <= {threshold}")
                return False
        else:
            logger.warning("Could not calculate drift scores. Retraining recommended.")
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
        if data is None or data.empty:
            logger.warning("Cannot calculate stats on empty data")
            return pd.DataFrame()
            
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            logger.warning("No numeric columns in data for statistics calculation")
            return pd.DataFrame()
        
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
        
        if not stats.empty:
            stats.to_csv(stats_path, index=False)
            logger.info(f"Data statistics saved to {stats_path}")
        else:
            logger.warning(f"Could not save data statistics to {stats_path} - empty stats dataframe")
    
    def train_lstm_model(self, data, ticker, seq_length=60, epochs=100, batch_size=32, min_data_points=100):
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
        min_data_points : int, optional
            Minimum number of data points required for training
            
        Returns:
        --------
        tuple : (trained model, scaler, features used)
        """
        logger.info(f"Training LSTM model for {ticker}...")
        
        # Validate input data
        if data is None or data.empty:
            logger.error(f"Cannot train model for {ticker}: Empty data")
            return None, None, None
        
        # Filter data for the specific ticker
        ticker_data = data[data['ticker'] == ticker].sort_values('Date') if 'ticker' in data.columns else data.sort_values('Date')
        
        # Check if enough data for training
        if len(ticker_data) < min_data_points:
            logger.warning(f"Insufficient data for {ticker}: {len(ticker_data)} < {min_data_points} required points")
            return None, None, None
        
        # Select features for LSTM
        base_features = ['Price', 'Open', 'High', 'Low', 'Vol.', 'Change %']
        available_features = [f for f in base_features if f in ticker_data.columns]
        
        # Add technical indicators if available
        technical_indicators = [col for col in ticker_data.columns if col.startswith('RSI') or 
                               col.startswith('MACD') or 
                               col.startswith('ATR') or 
                               col.startswith('SMA') or 
                               col.startswith('EMA')]
        
        # If we have at least 2 technical indicators, add them to features
        if len(technical_indicators) >= 2:
            available_features.extend(technical_indicators)
        
        # Ensure we have enough features
        if len(available_features) < 3:
            logger.warning(f"Not enough features for {ticker}, need at least 3 features. Available: {available_features}")
            
            # Try to derive additional features if Price is available
            if 'Price' in ticker_data.columns:
                logger.info(f"Attempting to derive additional features from Price for {ticker}")
                
                # Create a copy to avoid modifying the original
                ticker_data_enhanced = ticker_data.copy()
                
                # If Price exists but not other basic features, derive them
                if 'Price' in ticker_data_enhanced.columns:
                    for col in ['Open', 'High', 'Low']:
                        if col not in ticker_data_enhanced.columns:
                            ticker_data_enhanced[col] = ticker_data_enhanced['Price']
                    
                    # Add rolling metrics
                    if 'SMA_20' not in ticker_data_enhanced.columns:
                        ticker_data_enhanced['SMA_20'] = ticker_data_enhanced['Price'].rolling(window=20).mean()
                    
                    if 'SMA_50' not in ticker_data_enhanced.columns:
                        ticker_data_enhanced['SMA_50'] = ticker_data_enhanced['Price'].rolling(window=50).mean()
                    
                    if 'Volatility_20d' not in ticker_data_enhanced.columns:
                        ticker_data_enhanced['Volatility_20d'] = ticker_data_enhanced['Price'].rolling(window=20).std()
                    
                    # Recalculate available features
                    available_features = [f for f in base_features if f in ticker_data_enhanced.columns]
                    technical_indicators = [col for col in ticker_data_enhanced.columns if col.startswith('SMA') or col.startswith('Volatility')]
                    available_features.extend(technical_indicators)
                    
                    # Update ticker data to enhanced version
                    ticker_data = ticker_data_enhanced
                    
                    logger.info(f"Enhanced feature set for {ticker}: {available_features}")
            
            # Check again if we have enough features
            if len(available_features) < 3:
                logger.error(f"Still not enough features for {ticker} after enhancement. Cannot train model.")
                return None, None, None
        
        logger.info(f"Using features for {ticker}: {available_features}")
        
        # Extract features and handle NaN values
        X = ticker_data[available_features].values
        
        # Check for NaN values and handle them
        if np.isnan(X).any():
            logger.warning(f"NaN values found in input data for {ticker}. Filling with forward fill and mean.")
            # Create a DataFrame view for easier handling
            X_df = pd.DataFrame(X, columns=available_features)
            
            # Forward fill first, then backfill, then use mean for any remaining NaNs
            X_df = X_df.fillna(method='ffill').fillna(method='bfill')
            
            # If still have NaNs, use column means
            if X_df.isna().any().any():
                X_df = X_df.fillna(X_df.mean())
                
                # If still have NaNs after all this, drop those columns
                if X_df.isna().any().any():
                    nan_cols = X_df.columns[X_df.isna().any()].tolist()
                    logger.warning(f"Dropping columns with persistent NaNs for {ticker}: {nan_cols}")
                    
                    X_df = X_df.drop(columns=nan_cols)
                    available_features = [f for f in available_features if f not in nan_cols]
                    
                    if len(available_features) < 3:
                        logger.error(f"After dropping NaN columns, not enough features remain for {ticker}")
                        return None, None, None
            
            # Convert back to numpy array
            X = X_df.values
        
        # Scale the features
        try:
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            logger.error(f"Error scaling features for {ticker}: {str(e)}")
            return None, None, None
        
        # Create sequences for LSTM
        X_seq, y_seq = [], []
        
        for i in range(len(X_scaled) - seq_length):
            X_seq.append(X_scaled[i:i+seq_length])
            # Predict the next day's price (index 0 corresponds to 'Price')
            y_seq.append(X_scaled[i+seq_length, 0])
        
        if not X_seq:
            logger.error(f"Not enough data for {ticker} to create sequences. Need at least {seq_length+1} data points.")
            return None, None, None
        
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        
        # Verify shape is as expected
        if len(X_seq.shape) != 3:
            logger.error(f"Unexpected X_seq shape for {ticker}: {X_seq.shape}. Expected 3 dimensions.")
            return None, None, None
        
        # Split into train and validation sets
        train_size = int(len(X_seq) * 0.8)
        
        # Make sure we have enough data for both training and validation
        if train_size < 50 or (len(X_seq) - train_size) < 20:
            logger.warning(f"Limited data for {ticker} - adjusting split to ensure sufficient validation data")
            train_size = max(int(len(X_seq) * 0.7), len(X_seq) - 20)  # Ensure at least 20 validation samples
        
        X_train, X_val = X_seq[:train_size], X_seq[train_size:]
        y_train, y_val = y_seq[:train_size], y_seq[train_size:]
        
        logger.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
        
        # Build LSTM model with error handling
        try:
            # Check if inputs match expected shape
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
        except Exception as e:
            logger.error(f"Error building LSTM model for {ticker}: {str(e)}")
            return None, None, None
        
        # Train the model with early stopping and error handling
        try:
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            
            # Set batch size based on training data size
            adaptive_batch_size = min(batch_size, len(X_train) // 10) if len(X_train) > 10 else 1
            logger.info(f"Using batch size of {adaptive_batch_size} for {ticker}")
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=adaptive_batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],
                verbose=1
            )
            
            # Check if training was successful
            if np.isnan(history.history['loss'][-1]) or np.isnan(history.history['val_loss'][-1]):
                logger.error(f"Training failed for {ticker} - NaN loss values")
                return None, None, None
                
            # Evaluate the model
            val_loss = history.history['val_loss'][-1]
            logger.info(f"Validation loss for {ticker}: {val_loss:.4f}")
            
            # If validation loss is exceptionally high, model might be poor
            if val_loss > 0.2:  # This threshold may need adjustment
                logger.warning(f"High validation loss for {ticker}: {val_loss:.4f}")
        except Exception as e:
            logger.error(f"Error training LSTM model for {ticker}: {str(e)}")
            return None, None, None
        
        # Plot training history
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Training and Validation Loss for {ticker}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plot_path = os.path.join(self.models_dir, f"{ticker}_training_loss.png")
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Training history plot saved to {plot_path}")
        except Exception as e:
            logger.warning(f"Error creating training history plot for {ticker}: {str(e)}")
        
        # Save model and related artifacts
        try:
            # Save model
            model_path = os.path.join(self.models_dir, f"{ticker}_lstm.h5")
            model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save scaler
            scaler_path = os.path.join(self.models_dir, f"{ticker}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Scaler saved to {scaler_path}")
            
            # Save features used
            features_path = os.path.join(self.models_dir, f"{ticker}_features.pkl")
            with open(features_path, 'wb') as f:
                pickle.dump(available_features, f)
            logger.info(f"Features list saved to {features_path}")
            
            # Save model metadata
            metadata = {
                'ticker': ticker,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'features': available_features,
                'sequence_length': seq_length,
                'epochs': len(history.history['loss']),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }
            
            metadata_path = os.path.join(self.models_dir, f"{ticker}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
            
            return model, scaler, available_features
        except Exception as e:
            logger.error(f"Error saving model artifacts for {ticker}: {str(e)}")
            return model, scaler, available_features  # Return model even if saving failed
    
    def train_lstm_models(self, data, min_data_points=100):
        """
        Train LSTM models for all tickers in the data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data for model training
        min_data_points : int, optional
            Minimum number of data points required for a ticker to be trained
            
        Returns:
        --------
        dict : Dictionary mapping tickers to their trained models
        """
        if data is None or data.empty:
            logger.error("Cannot train models: Empty data")
            return {}
        
        # Check if we have a ticker column
        if 'ticker' not in data.columns:
            logger.warning("No ticker column in data. Adding default ticker.")
            data['ticker'] = 'DEFAULT'
        
        # Get unique tickers
        tickers = data['ticker'].unique()
        logger.info(f"Training models for {len(tickers)} tickers: {tickers}")
        
        models = {}
        successful_tickers = []
        failed_tickers = []
        
        for ticker in tickers:
            try:
                # Get ticker data
                ticker_data = data[data['ticker'] == ticker]
                
                # Check if we have enough data
                if len(ticker_data) < min_data_points:
                    logger.warning(f"Insufficient data for {ticker}: {len(ticker_data)} < {min_data_points}")
                    failed_tickers.append({'ticker': ticker, 'reason': 'insufficient_data'})
                    continue
                
                # Train the model
                model, scaler, features = self.train_lstm_model(data, ticker, min_data_points=min_data_points)
                
                if model is not None:
                    models[ticker] = {'model': model, 'scaler': scaler, 'features': features}
                    successful_tickers.append(ticker)
                else:
                    failed_tickers.append({'ticker': ticker, 'reason': 'training_failed'})
            except Exception as e:
                logger.error(f"Error training model for {ticker}: {str(e)}")
                failed_tickers.append({'ticker': ticker, 'reason': str(e)})
        
        logger.info(f"Successfully trained models for {len(successful_tickers)} tickers")
        if failed_tickers:
            logger.warning(f"Failed to train models for {len(failed_tickers)} tickers")
            
        return models
    
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
    
    def predict_next_n_days(self, model, scaler, last_sequence, n_days=7, feature_count=None):
        """
        Predict prices for the next n days.
        
        Parameters:
        -----------
        model : keras.Model
            Trained LSTM model
        scaler : MinMaxScaler
            Scaler used for feature normalization
        last_sequence : numpy.ndarray
            Last sequence used for prediction
        n_days : int, optional
            Number of days to predict
        feature_count : int, optional
            Number of features used in training
            
        Returns:
        --------
        list : Predicted prices for next n days
        """
        try:
            # Validate inputs
            if model is None or scaler is None or last_sequence is None:
                logger.error("Cannot predict: Missing model, scaler, or sequence data")
                return []
            
            if len(last_sequence.shape) != 3:
                logger.error(f"Invalid sequence shape: {last_sequence.shape}. Expected 3D array.")
                return []
            
            # Initialize with the last sequence
            curr_sequence = last_sequence.copy()
            predicted_prices = []
            
            for _ in range(n_days):
                # Predict next price
                try:
                    predicted_scaled = model.predict(curr_sequence)
                    
                    # Create a row with all features
                    if feature_count is None:
                        feature_count = curr_sequence.shape[2]
                    
                    # Use the last row as template and update the price (first column)
                    next_row = curr_sequence[0, -1, :].reshape(1, -1)
                    next_row[0, 0] = predicted_scaled[0, 0]
                    
                    # Inverse transform to get actual price
                    predicted_full_row = scaler.inverse_transform(next_row)
                    predicted_price = predicted_full_row[0, 0]
                    
                    predicted_prices.append(predicted_price)
                    
                    # Update sequence for next prediction
                    next_seq = np.roll(curr_sequence[0], -1, axis=0)
                    next_seq[-1] = next_row
                    curr_sequence = np.array([next_seq])
                except Exception as e:
                    logger.error(f"Error in prediction step: {str(e)}")
                    break
            
            return predicted_prices
        except Exception as e:
            logger.error(f"Error in predict_next_n_days: {str(e)}")
            return []
            
    def load_model(self, ticker):
        """
        Load a saved model for a specific ticker.
        
        Parameters:
        -----------
        ticker : str
            The ticker symbol to load model for
            
        Returns:
        --------
        tuple : (loaded model, scaler, features used)
        """
        try:
            # Paths for model artifacts
            model_path = os.path.join(self.models_dir, f"{ticker}_lstm.h5")
            scaler_path = os.path.join(self.models_dir, f"{ticker}_scaler.pkl")
            features_path = os.path.join(self.models_dir, f"{ticker}_features.pkl")
            
            # Check if files exist
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found for {ticker}: {model_path}")
                return None, None, None
                
            if not os.path.exists(scaler_path):
                logger.warning(f"Scaler file not found for {ticker}: {scaler_path}")
                return None, None, None
                
            if not os.path.exists(features_path):
                logger.warning(f"Features file not found for {ticker}: {features_path}")
                return None, None, None
            
            # Load model
            model = load_model(model_path)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load features
            with open(features_path, 'rb') as f:
                features = pickle.load(f)
                
            logger.info(f"Successfully loaded model for {ticker}")
            return model, scaler, features
        except Exception as e:
            logger.error(f"Error loading model for {ticker}: {str(e)}")
            return None, None, None 