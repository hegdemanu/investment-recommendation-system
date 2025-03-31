#!/usr/bin/env python3
"""
ML-based Trading Strategy
Uses predictions from ML models to generate trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
import requests
import json
from datetime import datetime, timedelta

from .base_strategy import BaseStrategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLStrategy(BaseStrategy):
    """Trading strategy based on ML model predictions"""
    
    def __init__(
        self,
        symbol: str,
        model_type: str = "ENSEMBLE",
        confidence_threshold: float = 0.02,
        stop_loss: float = 0.05,
        take_profit: float = 0.10,
        holding_period: int = 5,
        ml_service_url: str = "http://localhost:5001"
    ):
        """Initialize ML strategy"""
        name = f"ML_{model_type}_{symbol}"
        params = {
            "symbol": symbol,
            "model_type": model_type,
            "confidence_threshold": confidence_threshold,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "holding_period": holding_period
        }
        super().__init__(name, params)
        
        self.symbol = symbol
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.holding_period = holding_period
        self.ml_service_url = ml_service_url
        
        self.entry_price = None
        self.entry_date = None
        self.prediction_cache = {}
        self.last_prediction_date = None
    
    def get_predictions(self, date: datetime) -> Dict:
        """Get predictions from ML service"""
        # Check if predictions are in cache and still valid
        cache_key = f"{self.symbol}_{self.model_type}_{date.strftime('%Y-%m-%d')}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        try:
            # Call ML service API
            response = requests.post(
                f"{self.ml_service_url}/predict",
                json={
                    "symbol": self.symbol,
                    "modelType": self.model_type,
                    "horizon": self.holding_period
                }
            )
            
            if response.status_code == 200:
                prediction_data = response.json()
                self.prediction_cache[cache_key] = prediction_data
                self.last_prediction_date = date
                return prediction_data
            else:
                logger.error(f"Failed to get predictions: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return None
    
    def calculate_signal_strength(self, current_price: float, prediction_data: Dict) -> float:
        """Calculate signal strength based on predictions"""
        if not prediction_data or "predictions" not in prediction_data:
            return 0.0
        
        # Get the first prediction
        prediction = prediction_data["predictions"][0]
        predicted_price = prediction["price"]
        
        # Calculate expected return
        expected_return = (predicted_price - current_price) / current_price
        
        # Consider confidence from prediction
        confidence = prediction_data.get("confidence", 0.5)
        
        # Weighted signal strength
        signal_strength = expected_return * confidence
        
        return signal_strength
    
    def should_exit_position(self, data: pd.DataFrame) -> bool:
        """Determine if we should exit current position"""
        if self.position == 0 or self.entry_price is None:
            return False
        
        # Get current price
        current_price = data['Close'].iloc[-1]
        
        # Calculate current return
        current_return = (current_price - self.entry_price) / self.entry_price
        if self.position < 0:  # For short positions
            current_return = -current_return
        
        # Check stop loss
        if current_return <= -self.stop_loss:
            logger.info(f"Stop loss triggered: {current_return:.2%}")
            return True
        
        # Check take profit
        if current_return >= self.take_profit:
            logger.info(f"Take profit triggered: {current_return:.2%}")
            return True
        
        # Check holding period
        current_date = data.index[-1]
        if self.entry_date:
            days_held = (current_date - self.entry_date).days
            if days_held >= self.holding_period:
                logger.info(f"Holding period reached: {days_held} days")
                return True
        
        return False
    
    def generate_signal(self, data: pd.DataFrame) -> int:
        """Generate trading signal based on ML predictions"""
        if len(data) < 2:
            return 0
        
        # Get current price and date
        current_price = data['Close'].iloc[-1]
        current_date = data.index[-1]
        
        # Check if we should exit current position
        if self.position != 0 and self.should_exit_position(data):
            logger.info(f"Exiting position at {current_price}")
            self.entry_price = None
            self.entry_date = None
            return 0  # Signal to close position
        
        # If we have a position, maintain it unless exit conditions are met
        if self.position != 0:
            return self.position
        
        # Get predictions
        prediction_data = self.get_predictions(current_date)
        if not prediction_data:
            return 0
        
        # Calculate signal strength
        signal_strength = self.calculate_signal_strength(current_price, prediction_data)
        
        # Generate signal based on strength and threshold
        if signal_strength > self.confidence_threshold:
            # Bullish signal
            self.entry_price = current_price
            self.entry_date = current_date
            return 1
        elif signal_strength < -self.confidence_threshold:
            # Bearish signal
            self.entry_price = current_price
            self.entry_date = current_date
            return -1
        else:
            # Neutral
            return 0
    
    def get_next_prediction(self) -> Dict:
        """Get the next price prediction"""
        current_date = datetime.now()
        prediction_data = self.get_predictions(current_date)
        
        if not prediction_data or "predictions" not in prediction_data:
            return {
                "symbol": self.symbol,
                "predicted_price": None,
                "predicted_date": None,
                "signal": 0,
                "confidence": 0
            }
        
        # Get the first prediction
        prediction = prediction_data["predictions"][0]
        predicted_price = prediction["price"]
        predicted_date = prediction["date"]
        
        # Calculate signal strength
        current_price = self.get_current_price()
        signal_strength = self.calculate_signal_strength(current_price, prediction_data)
        
        # Determine signal
        if signal_strength > self.confidence_threshold:
            signal = 1
        elif signal_strength < -self.confidence_threshold:
            signal = -1
        else:
            signal = 0
        
        return {
            "symbol": self.symbol,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_date": predicted_date,
            "expected_return": (predicted_price - current_price) / current_price,
            "signal": signal,
            "signal_strength": signal_strength,
            "confidence": prediction_data.get("confidence", 0.5)
        }
    
    def get_current_price(self) -> float:
        """Get current price for the symbol"""
        try:
            # This is a simplified implementation
            # In a real system, you would use a market data provider
            response = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{self.symbol}?interval=1d&range=1d"
            )
            
            if response.status_code == 200:
                data = response.json()
                current_price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
                return current_price
            else:
                logger.error(f"Failed to get current price: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None 