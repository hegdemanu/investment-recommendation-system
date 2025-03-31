"""
Module for training and managing prediction models.
Supports LSTM, ARIMA-GARCH, and Prophet models.
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib
# Set the backend to non-interactive 'Agg' to avoid GUI issues
matplotlib.use('Agg')
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
        self.sequence_length = 60  # Default sequence length (~2 months of trading days)
        
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
    
    def train_lstm_model(self, data, ticker, seq_length=60, epochs=100, batch_size=32, min_data_points=180):
        """
        Train an LSTM model for a specific ticker using sliding window over 6 months data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data for model training
        ticker : str
            The ticker symbol to train model for
        seq_length : int, optional
            Sequence length for LSTM input, default 60 (~2 months of trading days)
        epochs : int, optional
            Number of training epochs
        batch_size : int, optional
            Batch size for training
        min_data_points : int, optional
            Minimum number of data points required for training (default 180 ~6 months)
            
        Returns:
        --------
        tuple : (trained model, scaler, features used)
        """
        logger.info(f"Training LSTM model for {ticker} with sliding window approach...")
        
        # Validate input data
        if data is None or data.empty:
            logger.error(f"Cannot train model for {ticker}: Empty data")
            return None, None, None
        
        # Filter data for the specific ticker - use at least 6 months of data (approx 180 days)
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
        
        # Define prediction horizons to test (1 to 30 days)
        prediction_horizons = [1, 3, 7, 14, 21, 30]
        max_horizon = 30
        
        # Create sequences for multi-horizon prediction
        X_seq, y_multi = [], []
        
        for i in range(len(X_scaled) - seq_length - max_horizon):
            # Input sequence
            X_seq.append(X_scaled[i:i+seq_length])
            
            # Create targets for multiple horizons
            targets = []
            for horizon in prediction_horizons:
                # Use the Price (first column) at each horizon
                targets.append(X_scaled[i+seq_length+horizon-1, 0])
            
            y_multi.append(targets)
        
        if not X_seq:
            logger.error(f"Not enough data for {ticker} to create sequences with horizons. Need at least {seq_length + max_horizon} data points.")
            return None, None, None
        
        X_seq = np.array(X_seq)
        y_multi = np.array(y_multi)
        
        logger.info(f"Created sequences with shape {X_seq.shape} and targets with shape {y_multi.shape}")
        
        # Split into train and validation sets
        train_size = int(len(X_seq) * 0.8)
        
        # Make sure we have enough data for both training and validation
        if train_size < 50 or (len(X_seq) - train_size) < 20:
            logger.warning(f"Limited data for {ticker} - adjusting split to ensure sufficient validation data")
            train_size = max(int(len(X_seq) * 0.7), len(X_seq) - 20)  # Ensure at least 20 validation samples
        
        X_train, X_val = X_seq[:train_size], X_seq[train_size:]
        y_train, y_val = y_multi[:train_size], y_multi[train_size:]
        
        logger.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
        
        # Build LSTM model with error handling
        try:
            # Check if inputs match expected shape
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(64, return_sequences=False),
                Dropout(0.2),
                Dense(len(prediction_horizons))  # Output one prediction for each horizon
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
            
            # Evaluate each prediction horizon
            val_predictions = model.predict(X_val)
            horizon_errors = []
            
            for i, horizon in enumerate(prediction_horizons):
                # Calculate RMSE for this horizon
                horizon_mse = np.mean((val_predictions[:, i] - y_val[:, i]) ** 2)
                horizon_rmse = np.sqrt(horizon_mse)
                horizon_errors.append((horizon, horizon_rmse))
                logger.info(f"Horizon {horizon} days - RMSE: {horizon_rmse:.4f}")
            
            # Find best horizon (lowest RMSE)
            horizon_errors.sort(key=lambda x: x[1])
            best_horizon = horizon_errors[0][0]
            best_horizon_rmse = horizon_errors[0][1]
            best_horizon_idx = prediction_horizons.index(best_horizon)
            
            logger.info(f"Best prediction horizon for {ticker}: {best_horizon} days with RMSE: {best_horizon_rmse:.4f}")
            
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
            
            # Plot prediction horizon errors
            plt.figure(figsize=(10, 6))
            horizons, errors = zip(*horizon_errors)
            plt.bar([str(h) for h in horizons], errors)
            plt.title(f'RMSE by Prediction Horizon for {ticker}')
            plt.xlabel('Prediction Horizon (days)')
            plt.ylabel('RMSE')
            plt.axvline(x=best_horizon_idx, color='r', linestyle='--', 
                        label=f'Best Horizon: {best_horizon} days')
            plt.xticks(rotation=45)
            plt.legend()
            horizon_plot_path = os.path.join(self.models_dir, f"{ticker}_horizon_errors.png")
            plt.savefig(horizon_plot_path)
            plt.close()
            logger.info(f"Horizon errors plot saved to {horizon_plot_path}")
        except Exception as e:
            logger.warning(f"Error creating plots for {ticker}: {str(e)}")
        
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
            
            # Save prediction horizons
            horizons_path = os.path.join(self.models_dir, f"{ticker}_horizons.pkl")
            with open(horizons_path, 'wb') as f:
                pickle.dump({
                    'prediction_horizons': prediction_horizons,
                    'best_horizon': best_horizon,
                    'best_horizon_idx': best_horizon_idx,
                    'horizon_errors': horizon_errors
                }, f)
            logger.info(f"Prediction horizons saved to {horizons_path}")
            
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
                'validation_samples': len(X_val),
                'prediction_horizons': prediction_horizons,
                'best_horizon': best_horizon,
                'best_horizon_rmse': float(best_horizon_rmse)
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
        dict : Dictionary with 'models' key mapping to trained models
        """
        if data is None or data.empty:
            logger.error("Cannot train models: Empty data")
            return {"models": {}}
        
        # Check if we have a ticker column
        if 'ticker' not in data.columns:
            logger.warning("No ticker column in data. Adding default ticker.")
            data['ticker'] = 'DEFAULT'
        
        # Get unique tickers
        tickers = data['ticker'].unique()
        logger.info(f"Training models for {len(tickers)} tickers: {tickers}")
        
        trained_models = {}
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
                    trained_models[ticker] = {'model': model, 'scaler': scaler, 'features': features}
                    successful_tickers.append(ticker)
                else:
                    failed_tickers.append({'ticker': ticker, 'reason': 'training_failed'})
            except Exception as e:
                logger.error(f"Error training model for {ticker}: {str(e)}")
                failed_tickers.append({'ticker': ticker, 'reason': str(e)})
        
        logger.info(f"Successfully trained models for {len(successful_tickers)} tickers")
        if failed_tickers:
            logger.warning(f"Failed to train models for {len(failed_tickers)} tickers")
            
        return {"models": trained_models, "scalers": {}, "features": {}}
    
    def load_lstm_models(self):
        """
        Load pre-trained LSTM models.
        
        Returns:
        --------
        dict : Dictionary with 'models' key mapping to loaded models with their scalers and features
        """
        result = {
            "models": {}
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
                
                result["models"][ticker] = {
                    'model': model,
                    'scaler': scaler,
                    'features': features
                }
                
                print(f"Successfully loaded model for {ticker}")
            except Exception as e:
                print(f"Error loading model for {ticker}: {str(e)}")
        
        print(f"Successfully loaded {len(result['models'])} models.")
        return result
    
    def predict_lstm(self, models, data):
        """
        Generate predictions using the trained LSTM models.
        
        Args:
            models (dict): Dictionary of trained LSTM models.
            data (pd.DataFrame): DataFrame containing the data to make predictions for.
            
        Returns:
            pd.DataFrame: DataFrame with original data and predictions.
        """
        if models is None or not models.get('models') or data is None or data.empty:
            logger.warning("No models or data available for prediction")
            return pd.DataFrame()
            
        # Create a copy of the data to avoid modifying the original
        predictions_df = data.copy()
        
        # Add prediction columns
        predictions_df['predicted_1d'] = np.nan
        predictions_df['predicted_3d'] = np.nan
        predictions_df['predicted_7d'] = np.nan
        predictions_df['predicted_14d'] = np.nan
        predictions_df['predicted_30d'] = np.nan
        
        # Group by ticker
        for ticker, ticker_data in predictions_df.groupby('ticker'):
            if ticker not in models['models']:
                logger.warning(f"No model found for ticker {ticker}")
                continue
                
            ticker_model = models['models'][ticker]
            model = ticker_model['model']
            scaler = ticker_model['scaler']
            features = ticker_model['features']
            
            # Get numerical features only
            numeric_data = ticker_data[features].copy()
            
            # Scale the data
            scaled_data = scaler.transform(numeric_data)
            
            # Create sequences
            X, _ = self._create_sequences(scaled_data, self.sequence_length)
            
            if len(X) == 0:
                logger.warning(f"Not enough data to create sequences for {ticker}")
                continue
                
            # Make predictions
            predictions = model.predict(X)
            
            # Adjust indices to match original dataframe
            start_idx = ticker_data.index[self.sequence_length]
            end_idx = ticker_data.index[self.sequence_length + len(predictions) - 1]
            
            pred_indices = predictions_df.loc[start_idx:end_idx].index
            
            # For each prediction horizon
            for i, days in enumerate([1, 3, 7, 14, 30]):
                col_name = f'predicted_{days}d'
                
                # Skip if we don't have enough predictions
                if i >= predictions.shape[1]:
                    continue
                    
                # Create a Series with predictions
                pred_series = pd.Series(predictions[:, i], index=pred_indices)
                
                # Update the dataframe
                predictions_df.loc[pred_indices, col_name] = pred_series
        
        return predictions_df
        
    def prepare_sequence_for_prediction(self, data, features, scaler):
        """
        Prepare the last sequence from the data for prediction.
        
        Args:
            data (pd.DataFrame): DataFrame containing the stock data.
            features (list): List of features used by the model.
            scaler (object): Fitted scaler used to normalize the data.
            
        Returns:
            tuple: (last_sequence, available_features)
                - last_sequence: The last sequence of data scaled and ready for prediction
                - available_features: List of features actually used (in case some were missing)
        """
        try:
            # Get the available features (some may be missing in the data)
            available_features = [f for f in features if f in data.columns]
            
            if not available_features:
                logger.error("No required features found in data")
                return None, []
                
            # Get numerical features only
            numeric_data = data[available_features].copy()
            
            # Handle NaN values (forward fill, then backward fill, then zero)
            numeric_data = numeric_data.ffill().bfill().fillna(0)
            
            # Scale the data
            scaled_data = scaler.transform(numeric_data)
            
            # Create the last sequence for prediction
            if len(scaled_data) < self.sequence_length:
                logger.warning(f"Not enough data points. Got {len(scaled_data)}, need {self.sequence_length}")
                return None, available_features
                
            last_sequence = scaled_data[-self.sequence_length:]
            last_sequence = np.expand_dims(last_sequence, axis=0)  # Add batch dimension
            
            return last_sequence, available_features
            
        except Exception as e:
            logger.error(f"Error preparing sequence: {str(e)}")
            return None, []
            
    def predict_future(self, model, last_sequence, scaler, features, days=30):
        """
        Predict future prices for a specified number of days.
        Uses the best prediction horizon determined during training.
        
        Args:
            model (keras.Model): Trained LSTM model.
            last_sequence (numpy.ndarray): The last sequence of data for prediction.
            scaler (object): The scaler used to normalize the data.
            features (list): List of features used in the model.
            days (int): Number of days to predict into the future.
            
        Returns:
            numpy.ndarray: Array of predicted prices.
        """
        if model is None or last_sequence is None or scaler is None:
            logger.error("Missing required components for prediction")
            return np.array([])
            
        try:
            # Try to load horizons information
            # Determine which ticker this model is for
            ticker = None
            for t in os.listdir(self.models_dir):
                if t.endswith("_lstm.h5") and os.path.exists(os.path.join(self.models_dir, t.replace("_lstm.h5", "_horizons.pkl"))):
                    # Check if this model file matches our model
                    if model == load_model(os.path.join(self.models_dir, t)):
                        ticker = t.replace("_lstm.h5", "")
                        break
            
            if ticker is None:
                logger.warning("Could not determine ticker for this model, using default prediction method")
                return self._predict_iterative(model, last_sequence, scaler, features, days)
            
            # Load horizons information
            horizons_path = os.path.join(self.models_dir, f"{ticker}_horizons.pkl")
            if not os.path.exists(horizons_path):
                logger.warning(f"Horizons information not found, using default prediction method")
                return self._predict_iterative(model, last_sequence, scaler, features, days)
            
            with open(horizons_path, 'rb') as f:
                horizons_info = pickle.load(f)
            
            prediction_horizons = horizons_info['prediction_horizons']
            best_horizon = horizons_info['best_horizon']
            best_horizon_idx = horizons_info['best_horizon_idx']
            
            logger.info(f"Using best prediction horizon of {best_horizon} days for {ticker}")
            
            # Get the price feature index
            price_idx = features.index('Price')
            
            # Make a copy of the last sequence
            sequence = last_sequence.copy()
            
            # Predictions to return
            predictions = []
            
            # Number of iterations needed to cover requested days
            iterations = (days + best_horizon - 1) // best_horizon
            days_predicted = 0
            
            for _ in range(iterations):
                if days_predicted >= days:
                    break
                
                # Get predictions for all horizons
                horizon_preds = model.predict(sequence)[0]
                
                # Get prediction for best horizon
                best_pred_scaled = horizon_preds[best_horizon_idx]
                
                # Create dummy array for inverse scaling
                dummy = np.zeros((1, len(features)))
                dummy[0, price_idx] = best_pred_scaled
                
                # Inverse transform to get actual price
                best_pred = scaler.inverse_transform(dummy)[0, price_idx]
                
                # Add to predictions
                predictions.append(best_pred)
                
                # Update sequence for next prediction
                # We'll shift by best_horizon days
                for _ in range(min(best_horizon, days - days_predicted)):
                    # Shift sequence one step left
                    sequence_shifted = sequence[0, 1:, :].copy()
                    
                    # Add new prediction as last entry
                    new_row = sequence_shifted[-1, :].copy()
                    new_row[price_idx] = best_pred_scaled
                    
                    sequence_with_new = np.vstack([sequence_shifted, new_row.reshape(1, -1)])
                    sequence[0] = sequence_with_new
                    
                    days_predicted += 1
                    
                    if days_predicted >= days:
                        break
            
            if len(predictions) > days:
                predictions = predictions[:days]
                
            return np.array(predictions)
                
        except Exception as e:
            logger.error(f"Error in predict_future: {str(e)}")
            logger.info("Falling back to default prediction method")
            return self._predict_iterative(model, last_sequence, scaler, features, days)
    
    def _predict_iterative(self, model, last_sequence, scaler, features, days=30):
        """
        Fallback method to predict future prices iteratively (day by day).
        
        Args:
            model (keras.Model): Trained LSTM model.
            last_sequence (numpy.ndarray): The last sequence of data for prediction.
            scaler (object): The scaler used to normalize the data.
            features (list): List of features used in the model.
            days (int): Number of days to predict into the future.
            
        Returns:
            numpy.ndarray: Array of predicted prices.
        """
        # Get the price feature index
        price_idx = features.index('Price')
        
        # Make a copy of the last sequence
        sequence = last_sequence.copy()
        
        # Store predictions
        predictions = []
        
        for _ in range(days):
            # Predict next day
            pred = model.predict(sequence)[0]
            
            # For multi-horizon model, use first output (1-day prediction)
            if isinstance(pred, np.ndarray) and pred.size > 1:
                pred = pred[0]
            
            # Save prediction
            predictions.append(pred)
            
            # Update sequence for next prediction
            # Shift sequence one step left
            sequence_shifted = sequence[0, 1:, :].copy()
            
            # Add new prediction as last entry
            new_row = sequence_shifted[-1, :].copy()
            new_row[price_idx] = pred
            
            sequence_with_new = np.vstack([sequence_shifted, new_row.reshape(1, -1)])
            sequence[0] = sequence_with_new
        
        # Convert predictions to original scale
        dummy = np.zeros((len(predictions), len(features)))
        dummy[:, price_idx] = predictions
        
        # Inverse transform
        unscaled_predictions = scaler.inverse_transform(dummy)[:, price_idx]
        
        return unscaled_predictions
    
    def backtracking_analysis(self, data, model, scaler, features, periods):
        """
        Perform backtracking analysis for a model on historical periods.
        
        Args:
            data (pd.DataFrame): DataFrame with historical data.
            model (keras.Model): Trained LSTM model.
            scaler (object): Scaler used to normalize data.
            features (list): Features used by the model.
            periods (dict): Dictionary with period names as keys and (start_date, end_date) tuples as values.
            
        Returns:
            dict: Dictionary with period names as keys and performance metrics as values.
        """
        if data is None or data.empty or model is None or scaler is None:
            logger.error("Missing required components for backtracking analysis")
            return {}
            
        try:
            # Ensure data has a datetime index or Date column
            if 'Date' in data.columns:
                data = data.copy()
                if not pd.api.types.is_datetime64_any_dtype(data['Date']):
                    data['Date'] = pd.to_datetime(data['Date'])
            else:
                logger.error("Date column not found in data")
                return {}
                
            results = {}
            
            for period_name, (start_date, end_date) in periods.items():
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                # Filter data for this period
                period_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()
                
                # Skip if not enough data
                if len(period_data) < self.sequence_length + 30:
                    logger.warning(f"Not enough data for period {period_name}")
                    continue
                    
                # Get available features
                available_features = [f for f in features if f in period_data.columns]
                
                if not available_features:
                    logger.warning(f"No required features in data for period {period_name}")
                    continue
                    
                # Prepare data
                numeric_data = period_data[available_features].copy()
                numeric_data = numeric_data.ffill().bfill().fillna(0)
                scaled_data = scaler.transform(numeric_data)
                
                # Create sequences
                X, _ = self._create_sequences(scaled_data, self.sequence_length)
                
                if len(X) == 0:
                    logger.warning(f"Not enough data to create sequences for period {period_name}")
                    continue
                    
                # Make predictions
                predictions = model.predict(X)
                
                # Extract actual prices for comparison
                price_idx = features.index('Price') if 'Price' in features else None
                if price_idx is None:
                    logger.warning("Price feature not found in model features")
                    continue
                    
                # For each prediction horizon
                period_results = {}
                
                for i, days in enumerate([1, 3, 7, 14, 30]):
                    # Skip if we don't have enough predictions or data
                    if i >= predictions.shape[1] or days >= len(period_data) - self.sequence_length:
                        continue
                        
                    # Get actual prices
                    actual_prices = []
                    for j in range(len(predictions)):
                        if self.sequence_length + j + days < len(period_data):
                            actual_prices.append(period_data['Price'].iloc[self.sequence_length + j + days])
                            
                    if not actual_prices:
                        continue
                        
                    # Create dummy arrays for inverse scaling
                    pred_dummy = np.zeros((len(predictions), len(features)))
                    pred_dummy[:, price_idx] = predictions[:, i]
                    
                    # Inverse transform
                    unscaled_predictions = scaler.inverse_transform(pred_dummy)[:, price_idx]
                    
                    # Compute metrics
                    actual_prices = np.array(actual_prices[:len(unscaled_predictions)])
                    
                    # Calculate metrics
                    mse = np.mean((unscaled_predictions - actual_prices) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(unscaled_predictions - actual_prices))
                    mape = np.mean(np.abs((actual_prices - unscaled_predictions) / actual_prices)) * 100
                    
                    # Coefficient of determination (R²)
                    ss_res = np.sum((actual_prices - unscaled_predictions) ** 2)
                    ss_tot = np.sum((actual_prices - np.mean(actual_prices)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Store results
                    period_results[f'{days}d'] = {
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'mape': float(mape),
                        'r2': float(r2),
                        'samples': len(actual_prices)
                    }
                
                # Add summary stats for the period
                if period_results:
                    # Get average metrics across horizons
                    avg_mse = np.mean([res['mse'] for _, res in period_results.items()])
                    avg_rmse = np.mean([res['rmse'] for _, res in period_results.items()])
                    avg_mae = np.mean([res['mae'] for _, res in period_results.items()])
                    avg_mape = np.mean([res['mape'] for _, res in period_results.items()])
                    avg_r2 = np.mean([res['r2'] for _, res in period_results.items()])
                    
                    period_results['summary'] = {
                        'avg_mse': float(avg_mse),
                        'avg_rmse': float(avg_rmse),
                        'avg_mae': float(avg_mae),
                        'avg_mape': float(avg_mape),
                        'avg_r2': float(avg_r2),
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'data_points': len(period_data)
                    }
                    
                    # Add trend analysis
                    period_prices = period_data['Price'].values
                    price_change = (period_prices[-1] / period_prices[0] - 1) * 100
                    
                    if price_change > 15:
                        trend = "Strong Uptrend"
                    elif price_change > 5:
                        trend = "Moderate Uptrend"
                    elif price_change > -5:
                        trend = "Sideways"
                    elif price_change > -15:
                        trend = "Moderate Downtrend"
                    else:
                        trend = "Strong Downtrend"
                        
                    period_results['summary']['trend'] = trend
                    period_results['summary']['price_change_pct'] = float(price_change)
                    
                    results[period_name] = period_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtracking_analysis: {str(e)}")
            return {}
    
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