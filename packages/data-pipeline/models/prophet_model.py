import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Dict, List, Any, Union, Tuple
import os
import pickle
import json
import matplotlib.pyplot as plt

class ProphetModel:
    """
    Prophet model for time series forecasting
    """
    
    def __init__(self, 
                model_params: Dict[str, Any] = None,
                cache_dir: str = "models/prophet/"):
        """
        Initialize Prophet model with optional parameters

        Args:
            model_params: Dictionary of parameters for Prophet model
            cache_dir: Directory to cache model files
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_params = model_params or {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'additive',
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet model
        
        Args:
            data: DataFrame with at least 'Date' and 'Close' columns
            
        Returns:
            DataFrame formatted for Prophet
        """
        if 'Date' not in data.columns or 'Close' not in data.columns:
            raise ValueError("Data must contain 'Date' and 'Close' columns")
        
        prophet_df = data[['Date', 'Close']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Add exogenous features if available
        self.exogenous_features = []
        for col in data.columns:
            if col not in ['Date', 'Close', 'ticker']:
                if pd.api.types.is_numeric_dtype(data[col]):
                    self.exogenous_features.append(col)
                    prophet_df[col] = data[col]
        
        return prophet_df
    
    def train(self, data: pd.DataFrame, ticker: str = None) -> Dict[str, Any]:
        """
        Train Prophet model
        
        Args:
            data: DataFrame with historical data
            ticker: Ticker symbol for saving model
            
        Returns:
            Training metrics
        """
        self.ticker = ticker
        self.logger.info(f"Training Prophet model for {ticker}")
        
        # Prepare data
        prophet_df = self._prepare_data(data)
        
        # Initialize and fit model
        self.model = Prophet(**self.model_params)
        
        # Add regressors for exogenous features
        for feature in self.exogenous_features:
            self.model.add_regressor(feature)
            
        self.model.fit(prophet_df)
        
        # Make in-sample predictions
        forecast = self.model.predict(prophet_df)
        
        # Calculate metrics
        y_true = prophet_df['y'].values
        y_pred = forecast['yhat'].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics = {
            'rmse': rmse,
            'mae': mae
        }
        
        self.logger.info(f"Prophet model training complete for {ticker}. RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Save model if ticker is provided
        if ticker:
            self.save(ticker)
            
            # Plot forecast components
            fig = self.model.plot_components(forecast)
            plt.savefig(os.path.join(self.cache_dir, f"{ticker}_components.png"))
            plt.close()
            
            # Plot forecast vs actual
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(prophet_df['ds'], prophet_df['y'], 'k.', label='Actual')
            ax.plot(forecast['ds'], forecast['yhat'], 'r-', label='Predicted')
            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.2)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(f'Prophet Forecast for {ticker}')
            ax.legend()
            plt.savefig(os.path.join(self.cache_dir, f"{ticker}_forecast.png"))
            plt.close()
            
        return metrics
    
    def predict(self, data: pd.DataFrame = None, periods: int = 30) -> Union[float, pd.DataFrame]:
        """
        Make predictions with trained model
        
        Args:
            data: Optional current data to use for prediction
            periods: Number of periods to forecast
            
        Returns:
            Forecast dataframe or single value for next period
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # If data is provided, use it for prediction
        if data is not None:
            prophet_df = self._prepare_data(data)
            
            # Make future dataframe including exogenous features from data
            future = prophet_df.copy()
        else:
            # Create future dataframe without exogenous features
            future = self.model.make_future_dataframe(periods=periods)
            
            # If we don't have exogenous features for future periods,
            # we can't make accurate predictions beyond the first period
            if self.exogenous_features and periods > 1:
                self.logger.warning("Making predictions without exogenous features for future periods")
        
        # Make predictions
        forecast = self.model.predict(future)
        
        # Return single value for next period if no data is provided
        if data is None and periods == 1:
            return forecast['yhat'].iloc[-1]
            
        return forecast
    
    def save(self, ticker: str) -> None:
        """
        Save model to disk
        
        Args:
            ticker: Ticker symbol
        """
        if self.model is None:
            raise ValueError("No trained model to save")
            
        # Create model directory if it doesn't exist
        model_dir = os.path.join(self.cache_dir, ticker)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Prophet model
        # Prophet models can't be directly pickled, so we use its export method
        model_path = os.path.join(model_dir, f"{ticker}_prophet.json")
        with open(model_path, 'w') as f:
            json.dump(self.model.to_json(), f)
            
        # Save exogenous features
        features_path = os.path.join(model_dir, f"{ticker}_features.pkl")
        with open(features_path, 'wb') as f:
            pickle.dump(self.exogenous_features, f)
            
        # Save model metadata
        metadata = {
            'model_type': 'prophet',
            'ticker': ticker,
            'model_params': self.model_params,
            'exogenous_features': self.exogenous_features,
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(model_dir, f"{ticker}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        self.logger.info(f"Prophet model saved for {ticker}")
    
    def load(self, ticker: str) -> bool:
        """
        Load model from disk
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            True if model loaded successfully
        """
        model_dir = os.path.join(self.cache_dir, ticker)
        model_path = os.path.join(model_dir, f"{ticker}_prophet.json")
        
        if not os.path.exists(model_path):
            self.logger.warning(f"No Prophet model found for {ticker}")
            return False
            
        try:
            # Load Prophet model
            with open(model_path, 'r') as f:
                model_json = json.load(f)
                
            self.model = Prophet.from_json(model_json)
            self.ticker = ticker
            
            # Load exogenous features
            features_path = os.path.join(model_dir, f"{ticker}_features.pkl")
            with open(features_path, 'rb') as f:
                self.exogenous_features = pickle.load(f)
                
            # Load metadata
            metadata_path = os.path.join(model_dir, f"{ticker}_metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_params = metadata['model_params']
                
            self.logger.info(f"Prophet model loaded for {ticker}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading Prophet model for {ticker}: {e}")
            return False 