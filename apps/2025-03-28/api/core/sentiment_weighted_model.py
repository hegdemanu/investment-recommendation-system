"""
Sentiment-Weighted Model Training module.

This module integrates sentiment analysis with time series prediction models:
- Adjusts training data weights based on sentiment scores
- Enhances prediction models with sentiment signals
- Provides evaluation metrics for sentiment-augmented models
"""
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os
import json
from typing import Dict, List, Union, Tuple, Optional, Any

# Import sentiment analyzer
from app.core.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)

class SentimentWeightedModel:
    """
    A model that incorporates sentiment analysis for improved predictions.
    """
    
    def __init__(self, model_type="lstm", sentiment_analyzer=None):
        """
        Initialize the sentiment-weighted model.
        
        Args:
            model_type (str): Model type ("lstm", "rf", "gbr", "linear").
            sentiment_analyzer (SentimentAnalyzer, optional): Pre-initialized sentiment analyzer.
        """
        self.model_type = model_type.lower()
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Initialize sentiment analyzer if not provided
        if sentiment_analyzer is None:
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
                logger.info("Initialized default sentiment analyzer")
            except Exception as e:
                logger.error(f"Error initializing sentiment analyzer: {str(e)}")
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = sentiment_analyzer
            
        # Model hyperparameters
        self.hyperparameters = {
            "lstm": {
                "units": 50,
                "dropout": 0.2,
                "recurrent_dropout": 0.2,
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001
            },
            "rf": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            },
            "gbr": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            },
            "linear": {}
        }
        
        # Training history
        self.history = None
        self.metrics = {}
        
    def _create_model(self, input_shape, output_shape=1):
        """
        Create a new model based on the specified type.
        
        Args:
            input_shape (tuple): Input shape for the model.
            output_shape (int): Output shape for the model.
            
        Returns:
            object: Initialized model.
        """
        if self.model_type == "lstm":
            # Create LSTM model
            hp = self.hyperparameters["lstm"]
            model = Sequential()
            model.add(LSTM(
                units=hp["units"],
                dropout=hp["dropout"],
                recurrent_dropout=hp["recurrent_dropout"],
                return_sequences=True,
                input_shape=input_shape
            ))
            model.add(LSTM(hp["units"] // 2, return_sequences=False))
            model.add(Dropout(hp["dropout"]))
            model.add(Dense(output_shape))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=hp["learning_rate"]),
                loss='mse'
            )
            
            return model
            
        elif self.model_type == "rf":
            # Create Random Forest model
            hp = self.hyperparameters["rf"]
            return RandomForestRegressor(
                n_estimators=hp["n_estimators"],
                max_depth=hp["max_depth"],
                min_samples_split=hp["min_samples_split"],
                min_samples_leaf=hp["min_samples_leaf"],
                random_state=42
            )
            
        elif self.model_type == "gbr":
            # Create Gradient Boosting model
            hp = self.hyperparameters["gbr"]
            return GradientBoostingRegressor(
                n_estimators=hp["n_estimators"],
                learning_rate=hp["learning_rate"],
                max_depth=hp["max_depth"],
                min_samples_split=hp["min_samples_split"],
                min_samples_leaf=hp["min_samples_leaf"],
                random_state=42
            )
            
        elif self.model_type == "linear":
            # Create Linear Regression model
            return LinearRegression()
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def _calculate_sentiment_weights(self, sentiments):
        """
        Calculate weights for training samples based on sentiment.
        
        Args:
            sentiments (list): List of sentiment dictionaries.
            
        Returns:
            numpy.ndarray: Array of weights for each sample.
        """
        # Extract sentiment scores
        weights = np.ones(len(sentiments))
        
        for i, sentiment in enumerate(sentiments):
            if isinstance(sentiment, dict) and "label" in sentiment:
                # Adjust weight based on sentiment label and confidence
                label = sentiment["label"]
                score = sentiment.get("score", 0.5)
                
                if label == "bullish":
                    weights[i] = 1.0 + score * 0.5  # Increase weight for bullish sentiment
                elif label == "bearish":
                    weights[i] = 1.0 + score * 0.3  # Slightly increase weight for bearish (still important)
                # Neutral sentiment keeps weight of 1.0
                
        return weights
        
    def _preprocess_news_data(self, news_data):
        """
        Preprocess news data to extract sentiment features.
        
        Args:
            news_data (list): List of news articles.
            
        Returns:
            pandas.DataFrame: DataFrame with sentiment features.
        """
        if self.sentiment_analyzer is None:
            logger.warning("Sentiment analyzer not available. Skipping sentiment extraction.")
            return pd.DataFrame()
            
        try:
            # Analyze sentiment in news
            results = self.sentiment_analyzer.analyze_financial_news(news_data)
            
            # Extract sentiment features
            sentiment_data = []
            for result in results:
                sentiment = result.get("sentiment", {})
                
                if isinstance(sentiment, dict) and "label" in sentiment:
                    # Convert sentiment label to numeric
                    label_value = 0.0  # neutral
                    if sentiment["label"] == "bullish":
                        label_value = 1.0
                    elif sentiment["label"] == "bearish":
                        label_value = -1.0
                        
                    # Get sentiment score
                    score = sentiment.get("score", 0.5)
                    
                    # Get individual probabilities
                    probs = sentiment.get("probabilities", {})
                    bullish_prob = probs.get("bullish", 0.33)
                    neutral_prob = probs.get("neutral", 0.34)
                    bearish_prob = probs.get("bearish", 0.33)
                    
                    # Get date from article
                    date = result.get("publishedAt", "")
                    if not date and isinstance(result, dict):
                        # Try to extract date from metadata
                        metadata = result.get("metadata", {})
                        date = metadata.get("date", "")
                        
                    # Create sentiment entry
                    entry = {
                        "date": date,
                        "sentiment_label": label_value,
                        "sentiment_score": score,
                        "bullish_prob": bullish_prob,
                        "neutral_prob": neutral_prob,
                        "bearish_prob": bearish_prob,
                        "sentiment_weighted_value": label_value * score
                    }
                    sentiment_data.append(entry)
                    
            # Create DataFrame
            df = pd.DataFrame(sentiment_data)
            
            # Handle date conversion if date column exists
            if "date" in df.columns and len(df) > 0:
                try:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                except Exception as e:
                    logger.warning(f"Error converting dates: {str(e)}")
                    
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing news data: {str(e)}")
            return pd.DataFrame()
            
    def _prepare_data_for_lstm(self, X, y, sequence_length=10):
        """
        Prepare data for LSTM model by creating sequences.
        
        Args:
            X (numpy.ndarray): Feature array.
            y (numpy.ndarray): Target array.
            sequence_length (int): Length of sequences for LSTM.
            
        Returns:
            tuple: (X_seq, y_seq) prepared for LSTM.
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
            
        return np.array(X_seq), np.array(y_seq)
        
    def train(self, price_data, news_data=None, target_col='close', 
              sequence_length=10, test_size=0.2, use_sentiment=True):
        """
        Train the model on price data with optional sentiment data.
        
        Args:
            price_data (pandas.DataFrame): DataFrame with price data.
            news_data (list, optional): List of news articles.
            target_col (str): Target column name.
            sequence_length (int): Sequence length for LSTM.
            test_size (float): Proportion of test data.
            use_sentiment (bool): Whether to use sentiment data.
            
        Returns:
            dict: Training metrics.
        """
        logger.info(f"Training {self.model_type} model with sentiment weighting")
        
        try:
            # Convert price_data to DataFrame if it's not already
            if not isinstance(price_data, pd.DataFrame):
                price_data = pd.DataFrame(price_data)
                
            # Extract target variable
            if target_col not in price_data.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
                
            y = price_data[target_col].values.reshape(-1, 1)
            
            # Extract features (use all columns except target by default)
            feature_cols = [col for col in price_data.columns if col != target_col]
            X = price_data[feature_cols].values
            
            # Extract sentiment features if news_data is provided
            sentiment_features = None
            if news_data is not None and use_sentiment and self.sentiment_analyzer is not None:
                sentiment_df = self._preprocess_news_data(news_data)
                
                if not sentiment_df.empty:
                    # Align sentiment data with price data by date
                    if isinstance(price_data.index, pd.DatetimeIndex) and isinstance(sentiment_df.index, pd.DatetimeIndex):
                        # Resample sentiment data to match price data frequency
                        sentiment_df = sentiment_df.resample('D').mean().dropna()
                        
                        # Forward-fill missing values (use last known sentiment)
                        sentiment_df = sentiment_df.reindex(price_data.index, method='ffill')
                        
                        # Extract sentiment features
                        sentiment_features = sentiment_df.values
                        
                        # Combine with price features
                        if sentiment_features is not None and len(sentiment_features) == len(X):
                            X = np.hstack((X, sentiment_features))
                    else:
                        logger.warning("Could not align sentiment data with price data (different index types)")
                        
            # Calculate sample weights based on sentiment
            sample_weights = None
            if news_data is not None and use_sentiment and self.sentiment_analyzer is not None:
                sentiments = []
                for article in news_data:
                    if isinstance(article, dict) and "sentiment" in article:
                        sentiments.append(article["sentiment"])
                    else:
                        # Analyze sentiment if not already present
                        try:
                            text = article.get("title", "") + " " + article.get("description", "")
                            if text.strip():
                                sentiment = self.sentiment_analyzer.analyze_text(text)
                                sentiments.append(sentiment)
                            else:
                                sentiments.append({"label": "neutral", "score": 0.5})
                        except Exception as e:
                            logger.error(f"Error analyzing sentiment: {str(e)}")
                            sentiments.append({"label": "neutral", "score": 0.5})
                            
                # Calculate weights if we have sentiment data
                if sentiments:
                    # If we have fewer sentiments than data points, repeat the last sentiment
                    while len(sentiments) < len(X):
                        sentiments.append(sentiments[-1] if sentiments else {"label": "neutral", "score": 0.5})
                        
                    # If we have more sentiments than data points, truncate
                    sentiments = sentiments[:len(X)]
                    
                    # Calculate weights
                    sample_weights = self._calculate_sentiment_weights(sentiments)
            
            # Scale features and target
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y)
            
            # Split data into train and test sets
            if self.model_type == "lstm":
                # Prepare sequences for LSTM
                X_seq, y_seq = self._prepare_data_for_lstm(X_scaled, y_scaled, sequence_length)
                
                # Split into train and test
                X_train, X_test, y_train, y_test = train_test_split(
                    X_seq, y_seq, test_size=test_size, shuffle=False
                )
                
                # Adjust sample weights for sequences if available
                if sample_weights is not None:
                    # Create sequence weights based on the last element in each sequence
                    seq_weights = np.array([sample_weights[i+sequence_length-1] for i in range(len(X_scaled) - sequence_length)])
                    
                    # Split weights into train and test
                    _, _, train_weights, _ = train_test_split(
                        X_seq, seq_weights, test_size=test_size, shuffle=False
                    )
                else:
                    train_weights = None
                    
                # Get input shape for LSTM
                input_shape = (X_train.shape[1], X_train.shape[2])
                
                # Create and train LSTM model
                self.model = self._create_model(input_shape)
                
                # Train with sample weights if available
                hp = self.hyperparameters["lstm"]
                history = self.model.fit(
                    X_train, y_train,
                    epochs=hp["epochs"],
                    batch_size=hp["batch_size"],
                    validation_data=(X_test, y_test),
                    sample_weight=train_weights,
                    verbose=0
                )
                
                # Store training history
                self.history = {
                    "loss": history.history["loss"],
                    "val_loss": history.history["val_loss"]
                }
                
                # Make predictions
                y_pred_scaled = self.model.predict(X_test)
                
            else:
                # Split into train and test for other models
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_scaled.ravel(), test_size=test_size, shuffle=False
                )
                
                # Adjust sample weights if available
                if sample_weights is not None:
                    _, _, train_weights, _ = train_test_split(
                        X_scaled, sample_weights, test_size=test_size, shuffle=False
                    )
                else:
                    train_weights = None
                    
                # Create and train model
                self.model = self._create_model(X_train.shape[1])
                
                # Train with sample weights if available
                if hasattr(self.model, 'fit') and callable(getattr(self.model, 'fit')):
                    if train_weights is not None and hasattr(self.model, 'sample_weight'):
                        self.model.fit(X_train, y_train, sample_weight=train_weights)
                    else:
                        self.model.fit(X_train, y_train)
                        
                # Make predictions
                y_pred_scaled = self.model.predict(X_test).reshape(-1, 1)
                
            # Inverse transform predictions
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            y_true = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Store metrics
            self.metrics = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
                "test_size": test_size,
                "sequence_length": sequence_length if self.model_type == "lstm" else None,
                "used_sentiment": use_sentiment and news_data is not None
            }
            
            logger.info(f"Training completed with RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def predict(self, features, sequence_length=10):
        """
        Make predictions with the trained model.
        
        Args:
            features (numpy.ndarray or pandas.DataFrame): Input features.
            sequence_length (int): Sequence length for LSTM.
            
        Returns:
            numpy.ndarray: Predicted values.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        try:
            # Convert to numpy array if DataFrame
            if isinstance(features, pd.DataFrame):
                features = features.values
                
            # Scale features
            features_scaled = self.scaler_X.transform(features)
            
            # Make predictions based on model type
            if self.model_type == "lstm":
                # Prepare sequences for LSTM
                sequences = []
                for i in range(len(features_scaled) - sequence_length + 1):
                    sequences.append(features_scaled[i:i+sequence_length])
                    
                # Convert to numpy array
                if sequences:
                    sequences = np.array(sequences)
                    
                    # Make predictions
                    predictions_scaled = self.model.predict(sequences)
                    
                    # Inverse transform predictions
                    predictions = self.scaler_y.inverse_transform(predictions_scaled)
                    
                    return predictions
                else:
                    raise ValueError(f"Not enough data points for LSTM prediction (need at least {sequence_length})")
            else:
                # Make predictions for other models
                predictions_scaled = self.model.predict(features_scaled).reshape(-1, 1)
                
                # Inverse transform predictions
                predictions = self.scaler_y.inverse_transform(predictions_scaled)
                
                return predictions
                
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
            
    def get_sentiment_impact(self, price_data, news_data, target_col='close', sequence_length=10, test_size=0.2):
        """
        Assess the impact of sentiment on model performance.
        
        Args:
            price_data (pandas.DataFrame): DataFrame with price data.
            news_data (list): List of news articles.
            target_col (str): Target column name.
            sequence_length (int): Sequence length for LSTM.
            test_size (float): Proportion of test data.
            
        Returns:
            dict: Comparison of model performance with and without sentiment.
        """
        try:
            # Train model with sentiment
            metrics_with_sentiment = self.train(
                price_data, news_data, target_col, sequence_length, test_size, use_sentiment=True
            )
            
            # Save the model and metrics
            model_with_sentiment = self.model
            history_with_sentiment = self.history
            
            # Reset model
            self.model = None
            self.history = None
            
            # Train model without sentiment
            metrics_without_sentiment = self.train(
                price_data, news_data, target_col, sequence_length, test_size, use_sentiment=False
            )
            
            # Save the model and metrics
            model_without_sentiment = self.model
            history_without_sentiment = self.history
            
            # Restore the better model (lower RMSE)
            if metrics_with_sentiment["rmse"] <= metrics_without_sentiment["rmse"]:
                self.model = model_with_sentiment
                self.history = history_with_sentiment
                self.metrics = metrics_with_sentiment
            else:
                self.model = model_without_sentiment
                self.history = history_without_sentiment
                self.metrics = metrics_without_sentiment
                
            # Calculate impact
            rmse_diff = metrics_without_sentiment["rmse"] - metrics_with_sentiment["rmse"]
            rmse_pct_improvement = (rmse_diff / metrics_without_sentiment["rmse"]) * 100 if metrics_without_sentiment["rmse"] > 0 else 0
            
            r2_diff = metrics_with_sentiment["r2"] - metrics_without_sentiment["r2"]
            
            # Return comparison
            return {
                "with_sentiment": metrics_with_sentiment,
                "without_sentiment": metrics_without_sentiment,
                "rmse_improvement": float(rmse_diff),
                "rmse_pct_improvement": float(rmse_pct_improvement),
                "r2_improvement": float(r2_diff),
                "better_model": "with_sentiment" if metrics_with_sentiment["rmse"] <= metrics_without_sentiment["rmse"] else "without_sentiment"
            }
            
        except Exception as e:
            logger.error(f"Error assessing sentiment impact: {str(e)}")
            raise
            
    def save(self, filepath):
        """
        Save the model and related data to disk.
        
        Args:
            filepath (str): Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model based on type
            if self.model_type == "lstm":
                # Save Keras model
                model_path = f"{filepath}_model.h5"
                self.model.save(model_path)
            else:
                # Save scikit-learn model
                model_path = f"{filepath}_model.joblib"
                joblib.dump(self.model, model_path)
                
            # Save scalers
            scaler_x_path = f"{filepath}_scaler_x.joblib"
            scaler_y_path = f"{filepath}_scaler_y.joblib"
            joblib.dump(self.scaler_X, scaler_x_path)
            joblib.dump(self.scaler_y, scaler_y_path)
            
            # Save metadata
            metadata = {
                "model_type": self.model_type,
                "model_path": model_path,
                "scaler_x_path": scaler_x_path,
                "scaler_y_path": scaler_y_path,
                "hyperparameters": self.hyperparameters[self.model_type],
                "metrics": self.metrics,
                "history": self.history
            }
            
            metadata_path = f"{filepath}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Model saved to {filepath}")
            
            return metadata_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
            
    @classmethod
    def load(cls, filepath, sentiment_analyzer=None):
        """
        Load a model from disk.
        
        Args:
            filepath (str): Path to load the model from.
            sentiment_analyzer (SentimentAnalyzer, optional): Pre-initialized sentiment analyzer.
            
        Returns:
            SentimentWeightedModel: Loaded model.
        """
        try:
            # Load metadata
            metadata_path = f"{filepath}_metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                
            # Create instance
            model_type = metadata["model_type"]
            instance = cls(model_type=model_type, sentiment_analyzer=sentiment_analyzer)
            
            # Load scalers
            instance.scaler_X = joblib.load(metadata["scaler_x_path"])
            instance.scaler_y = joblib.load(metadata["scaler_y_path"])
            
            # Load model based on type
            if model_type == "lstm":
                # Load Keras model
                instance.model = tf.keras.models.load_model(metadata["model_path"])
            else:
                # Load scikit-learn model
                instance.model = joblib.load(metadata["model_path"])
                
            # Load metrics and history
            instance.metrics = metadata.get("metrics", {})
            instance.history = metadata.get("history", None)
            
            # Load hyperparameters
            if "hyperparameters" in metadata:
                instance.hyperparameters[model_type] = metadata["hyperparameters"]
                
            logger.info(f"Model loaded from {filepath}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start="2023-01-01", end="2023-03-01", freq="D")
    prices = np.random.normal(100, 5, len(dates)) + np.linspace(0, 10, len(dates))
    volumes = np.random.normal(1000000, 200000, len(dates))
    
    # Create DataFrame
    price_data = pd.DataFrame({
        "date": dates,
        "close": prices,
        "volume": volumes
    }).set_index("date")
    
    # Create sample news data
    news_data = [
        {
            "title": "Company reports strong earnings",
            "description": "Quarterly results exceed expectations with 15% growth",
            "publishedAt": "2023-01-15"
        },
        {
            "title": "Market experiences volatility",
            "description": "Investors cautious amid uncertain economic conditions",
            "publishedAt": "2023-02-01"
        },
        {
            "title": "Positive outlook ahead",
            "description": "Analysts project continued growth in the sector",
            "publishedAt": "2023-02-15"
        }
    ]
    
    # Create and train model
    model = SentimentWeightedModel(model_type="rf")
    
    # Assess sentiment impact
    impact = model.get_sentiment_impact(price_data, news_data)
    
    # Print results
    print(f"RMSE with sentiment: {impact['with_sentiment']['rmse']:.4f}")
    print(f"RMSE without sentiment: {impact['without_sentiment']['rmse']:.4f}")
    print(f"Improvement: {impact['rmse_pct_improvement']:.2f}%")
    print(f"Better model: {impact['better_model']}") 