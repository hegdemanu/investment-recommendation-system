import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from typing import Dict, List, Union, Any, Tuple
import logging
import os
import json
from datetime import datetime, timedelta

class ProphetForecaster:
    """
    Time series forecasting model using Facebook Prophet
    """
    
    def __init__(self, cache_dir: str = "trading_engine/models/prophet/"):
        """
        Initialize Prophet forecasting model
        
        Args:
            cache_dir: Directory to cache model files
        """
        self.logger = self._setup_logger()
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize model parameters
        self.default_params = {
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "holidays_prior_scale": 10.0,
            "changepoint_range": 0.8,
            "daily_seasonality": False,
            "weekly_seasonality": True,
            "yearly_seasonality": True
        }
        
        self.logger.info("ProphetForecaster initialized successfully")
    
    def _setup_logger(self):
        """Set up a logger for the ProphetForecaster"""
        logger = logging.getLogger("ProphetForecaster")
        logger.setLevel(logging.INFO)
        
        # Check if handler already exists to avoid duplicate logs
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def prepare_data(self, data: pd.DataFrame, target_col: str = 'Close') -> pd.DataFrame:
        """
        Prepare data for Prophet model
        
        Args:
            data: DataFrame with date column and price data
            target_col: Column name for the target variable
            
        Returns:
            DataFrame formatted for Prophet
        """
        # Prophet requires columns named 'ds' and 'y'
        prophet_data = data.reset_index() if data.index.name == 'Date' or isinstance(data.index, pd.DatetimeIndex) else data.copy()
        
        # Find date column
        date_col = None
        for col in prophet_data.columns:
            if col.lower() in ['date', 'datetime', 'timestamp']:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError("Date column not found in data")
        
        # Format data for Prophet
        prophet_data = prophet_data[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
        
        # Ensure datetime format
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        
        return prophet_data
    
    def train(self, data: pd.DataFrame, target_col: str = 'Close', params: Dict = None) -> Dict:
        """
        Train Prophet model on historical data
        
        Args:
            data: DataFrame with date column and price data
            target_col: Column name for the target variable
            params: Dictionary of model parameters
            
        Returns:
            Dictionary with training results
        """
        try:
            # Prepare data
            prophet_data = self.prepare_data(data, target_col)
            
            # Set model parameters
            model_params = self.default_params.copy()
            if params:
                model_params.update(params)
            
            # Initialize and train model
            self.logger.info("Training Prophet model...")
            self.model = Prophet(**model_params)
            
            # Add country-specific holidays
            self.model.add_country_holidays(country_name='US')
            
            # Fit model
            self.model.fit(prophet_data)
            
            self.logger.info("Prophet model trained successfully")
            
            # Save model parameters
            model_info = {
                "target_column": target_col,
                "params": model_params,
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_points": len(prophet_data)
            }
            
            return {
                "success": True,
                "model_info": model_info
            }
            
        except Exception as e:
            self.logger.error(f"Error training Prophet model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict(self, periods: int = 30, include_history: bool = True) -> pd.DataFrame:
        """
        Generate forecasts for future periods
        
        Args:
            periods: Number of periods to forecast
            include_history: Whether to include historical data in the forecast
            
        Returns:
            DataFrame with forecast results
        """
        try:
            if not hasattr(self, 'model'):
                raise ValueError("Model not trained. Call train() first.")
                
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, include_history=include_history)
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error generating forecast: {e}")
            return pd.DataFrame()
    
    def evaluate(self, test_data: pd.DataFrame, target_col: str = 'Close') -> Dict:
        """
        Evaluate model performance on test data
        
        Args:
            test_data: DataFrame with actual values
            target_col: Column name for the target variable
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if not hasattr(self, 'model'):
                raise ValueError("Model not trained. Call train() first.")
                
            # Prepare test data
            test_prophet = self.prepare_data(test_data, target_col)
            
            # Generate predictions for test period
            predictions = self.predict(periods=0, include_history=True)
            
            # Merge predictions with actual values
            eval_df = pd.merge(test_prophet, predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
            
            # Calculate metrics
            mse = np.mean((eval_df['y'] - eval_df['yhat'])**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(eval_df['y'] - eval_df['yhat']))
            mape = np.mean(np.abs((eval_df['y'] - eval_df['yhat']) / eval_df['y'])) * 100
            
            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "mape": mape
            }
            
            return {
                "success": True,
                "metrics": metrics,
                "predictions": eval_df
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def plot_forecast(self, forecast: pd.DataFrame = None, figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot forecast results
        
        Args:
            forecast: DataFrame with forecast results (from predict())
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure object
        """
        try:
            if not hasattr(self, 'model'):
                raise ValueError("Model not trained. Call train() first.")
                
            if forecast is None:
                forecast = self.predict()
                
            # Create figure
            fig = plt.figure(figsize=figsize)
            
            # Plot forecast
            self.model.plot(forecast, fig=fig)
            plt.title('Prophet Model Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting forecast: {e}")
            fig = plt.figure(figsize=figsize)
            plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            return fig
    
    def plot_components(self, forecast: pd.DataFrame = None, figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """
        Plot forecast components (trend, seasonality, etc.)
        
        Args:
            forecast: DataFrame with forecast results (from predict())
            figsize: Figure size as (width, height)
            
        Returns:
            Matplotlib figure object
        """
        try:
            if not hasattr(self, 'model'):
                raise ValueError("Model not trained. Call train() first.")
                
            if forecast is None:
                forecast = self.predict()
                
            # Create figure
            fig = plt.figure(figsize=figsize)
            
            # Plot components
            self.model.plot_components(forecast, fig=fig)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting components: {e}")
            fig = plt.figure(figsize=figsize)
            plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            return fig
    
    def save_model(self, path: str = None, ticker: str = None):
        """
        Save model to JSON file
        
        Args:
            path: Path to save model
            ticker: Ticker symbol for model (used in filename)
        """
        try:
            if not hasattr(self, 'model'):
                raise ValueError("Model not trained. Call train() first.")
                
            if path is None:
                if ticker:
                    path = os.path.join(self.cache_dir, f"{ticker}_prophet_model.json")
                else:
                    path = os.path.join(self.cache_dir, "prophet_model.json")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            with open(path, 'w') as f:
                json.dump(self.model.to_json(), f)
                
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self, path: str):
        """
        Load model from JSON file
        
        Args:
            path: Path to model file
        """
        try:
            with open(path, 'r') as f:
                model_json = json.load(f)
                
            self.model = Prophet.from_json(model_json)
            self.logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise 