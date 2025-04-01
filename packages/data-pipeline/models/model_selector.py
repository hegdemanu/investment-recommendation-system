import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Callable
from enum import Enum

from trading_engine.models.lstm_model import LSTMModel
from trading_engine.models.arima_model import ARIMAModel
from trading_engine.models.prophet_model import ProphetModel


class ModelType(Enum):
    LSTM = "lstm"
    ARIMA = "arima"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


class ModelSelector:
    """
    Dynamic model selector that switches between different models based on performance
    """
    
    def __init__(self):
        """
        Initialize the model selector
        """
        self.models = {}
        self.performance_history = {}
        self.current_best_model = None
        
    def add_model(self, model_type: ModelType, model: Any, name: str = None) -> None:
        """
        Add a model to the selector
        
        Args:
            model_type: Type of the model
            model: The model instance
            name: Optional name for the model
        """
        if name is None:
            name = f"{model_type.value}_{len(self.models) + 1}"
            
        self.models[name] = {
            "type": model_type,
            "model": model,
            "performance": []
        }
        
    def evaluate_models(self, test_data: pd.DataFrame, metric_func: Callable[[np.ndarray, np.ndarray], float]) -> Dict[str, float]:
        """
        Evaluate all models on test data using the provided metric function
        
        Args:
            test_data: Test data to evaluate on
            metric_func: Function to compute the metric (e.g., RMSE, MAE)
            
        Returns:
            Dictionary of model names and their metrics
        """
        metrics = {}
        
        for name, model_info in self.models.items():
            model = model_info["model"]
            model_type = model_info["type"]
            
            # Get predictions based on model type
            if model_type == ModelType.LSTM:
                pred = model.predict(test_data)
            elif model_type == ModelType.ARIMA:
                # For ARIMA, we just need the last prediction
                pred = model.predict(1)[0]
            elif model_type == ModelType.PROPHET:
                # For Prophet, use the provided data for prediction
                forecast = model.predict(test_data)
                # Extract the last prediction
                pred = forecast['yhat'].iloc[-1]
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Compute metric
            actual = test_data["Close"].values[-1]
            metric_value = metric_func(np.array([actual]), np.array([pred]))
            
            # Store metric
            metrics[name] = metric_value
            self.models[name]["performance"].append(metric_value)
            
        # Update performance history
        self.performance_history[len(self.performance_history)] = metrics
        
        # Update best model
        self._update_best_model()
        
        return metrics
    
    def _update_best_model(self) -> None:
        """
        Update the current best model based on recent performance
        """
        if not self.performance_history:
            return
        
        # Get latest performance
        latest_metrics = self.performance_history[max(self.performance_history.keys())]
        
        # Find model with lowest metric (assuming lower is better, like RMSE or MAE)
        best_model_name = min(latest_metrics, key=latest_metrics.get)
        self.current_best_model = best_model_name
    
    def predict(self, data: pd.DataFrame) -> Union[float, np.ndarray]:
        """
        Make a prediction using the best model
        
        Args:
            data: Input data
            
        Returns:
            Prediction
        """
        if self.current_best_model is None:
            # If no best model yet, use the first one
            if not self.models:
                raise ValueError("No models available for prediction.")
            self.current_best_model = list(self.models.keys())[0]
        
        model_info = self.models[self.current_best_model]
        model = model_info["model"]
        model_type = model_info["type"]
        
        # Get prediction based on model type
        if model_type == ModelType.LSTM:
            return model.predict(data)
        elif model_type == ModelType.ARIMA:
            return model.predict(1)[0]
        elif model_type == ModelType.PROPHET:
            forecast = model.predict(data)
            return forecast['yhat'].iloc[-1]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def get_best_model(self) -> Dict[str, Any]:
        """
        Get the current best model
        
        Returns:
            Dictionary with best model information
        """
        if self.current_best_model is None:
            if not self.models:
                raise ValueError("No models available.")
            self.current_best_model = list(self.models.keys())[0]
        
        return {
            "name": self.current_best_model,
            "type": self.models[self.current_best_model]["type"],
            "model": self.models[self.current_best_model]["model"],
            "performance": self.models[self.current_best_model]["performance"]
        }
    
    def get_all_model_performances(self) -> Dict[str, List[float]]:
        """
        Get performance history for all models
        
        Returns:
            Dictionary of model names and their performance history
        """
        performances = {}
        for name, model_info in self.models.items():
            performances[name] = model_info["performance"]
        
        return performances 