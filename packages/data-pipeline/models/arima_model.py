import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import List, Tuple, Dict, Any, Optional
import pickle


class ARIMAModel:
    """
    ARIMA model for time series prediction
    """
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        """
        Initialize the ARIMA model
        
        Args:
            order: ARIMA order parameters (p, d, q)
                p: autoregressive order
                d: differencing order
                q: moving average order
        """
        self.order = order
        self.model = None
        self.model_fit = None
        
    def train(self, data: pd.DataFrame, column: str = 'Close') -> Dict[str, Any]:
        """
        Train the ARIMA model
        
        Args:
            data: DataFrame with time series data
            column: Column name to use for training
            
        Returns:
            Dictionary with model summary
        """
        # Extract the time series
        series = data[column].values
        
        # Create ARIMA model
        self.model = ARIMA(series, order=self.order)
        
        # Fit the model
        self.model_fit = self.model.fit()
        
        return {
            'aic': self.model_fit.aic,
            'bic': self.model_fit.bic,
            'summary': self.model_fit.summary()
        }
    
    def predict(self, forecast_steps: int = 1) -> List[float]:
        """
        Make predictions using the trained model
        
        Args:
            forecast_steps: Number of steps to forecast
            
        Returns:
            List of predicted values
        """
        if self.model_fit is None:
            raise ValueError("Model is not trained. Call train() first.")
        
        # Make forecast
        forecast = self.model_fit.forecast(steps=forecast_steps)
        
        return forecast.tolist()
    
    def update(self, new_data: np.ndarray) -> None:
        """
        Update the model with new observations
        
        Args:
            new_data: New observations to add to the model
        """
        if self.model_fit is None:
            raise ValueError("Model is not trained. Call train() first.")
        
        # Update model with new data
        self.model_fit = self.model_fit.append(new_data)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the model to disk
        
        Args:
            filepath: Path to save the model
        """
        if self.model_fit is None:
            raise ValueError("Model is not trained. Call train() first.")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ARIMAModel':
        """
        Load the model from disk
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded ARIMAModel instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    @staticmethod
    def find_best_parameters(data: pd.DataFrame, column: str = 'Close', 
                           p_range: Tuple[int, int] = (0, 5),
                           d_range: Tuple[int, int] = (0, 2),
                           q_range: Tuple[int, int] = (0, 5)) -> Dict[str, Any]:
        """
        Find the best ARIMA parameters using AIC criterion
        
        Args:
            data: DataFrame with time series data
            column: Column name to use
            p_range: Range of p values to try (min, max)
            d_range: Range of d values to try (min, max)
            q_range: Range of q values to try (min, max)
            
        Returns:
            Dictionary with best parameters and AIC
        """
        series = data[column].values
        best_aic = float('inf')
        best_order = None
        best_model = None
        
        for p in range(p_range[0], p_range[1] + 1):
            for d in range(d_range[0], d_range[1] + 1):
                for q in range(q_range[0], q_range[1] + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        model_fit = model.fit()
                        
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_order = (p, d, q)
                            best_model = model_fit
                    except:
                        continue
        
        if best_model is None:
            raise ValueError("Could not find a suitable ARIMA model")
        
        # Create and return the best model
        best_arima = ARIMAModel(order=best_order)
        best_arima.model = ARIMA(series, order=best_order)
        best_arima.model_fit = best_model
        
        return {
            'model': best_arima,
            'order': best_order,
            'aic': best_aic
        } 