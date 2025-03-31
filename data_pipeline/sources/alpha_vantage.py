import requests
import pandas as pd
from typing import Dict, Any, Optional
import time
from datetime import datetime, timedelta
import os


class AlphaVantageAPI:
    """
    API client for Alpha Vantage market data
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the Alpha Vantage API client
        
        Args:
            api_key: Alpha Vantage API key (if not provided, will look for it in environment variable)
        """
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required. Set it in the constructor or as an environment variable.")
        
        self.base_url = "https://www.alphavantage.co/query"
    
    def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """
        Make a request to the Alpha Vantage API
        
        Args:
            params: Request parameters
            
        Returns:
            JSON response
        """
        # Add API key to params
        params["apikey"] = self.api_key
        
        # Make request
        response = requests.get(self.base_url, params=params)
        
        # Check for errors
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        data = response.json()
        
        # Check for API error messages
        if "Error Message" in data:
            raise Exception(f"API error: {data['Error Message']}")
        
        return data
    
    def get_daily_adjusted(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        """
        Get daily adjusted time series for a symbol
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            outputsize: "compact" (last 100 data points) or "full" (up to 20 years of data)
            
        Returns:
            DataFrame with daily adjusted time series
        """
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize
        }
        
        data = self._make_request(params)
        
        # Parse time series data
        time_series = data.get("Time Series (Daily)", {})
        
        if not time_series:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        
        # Convert columns to proper types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Rename columns
        df.columns = [col.split(". ")[1] for col in df.columns]
        
        # Sort by date (ascending)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
    
    def get_intraday(self, symbol: str, interval: str = "5min", outputsize: str = "compact") -> pd.DataFrame:
        """
        Get intraday time series for a symbol
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            interval: Time interval between data points (1min, 5min, 15min, 30min, 60min)
            outputsize: "compact" (last 100 data points) or "full" (up to 20 years of data)
            
        Returns:
            DataFrame with intraday time series
        """
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize
        }
        
        data = self._make_request(params)
        
        # Parse time series data
        time_series_key = f"Time Series ({interval})"
        time_series = data.get(time_series_key, {})
        
        if not time_series:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        
        # Convert columns to proper types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Rename columns
        df.columns = [col.split(". ")[1] for col in df.columns]
        
        # Sort by date (ascending)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        return df
    
    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get company overview for a symbol
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            
        Returns:
            Dictionary with company information
        """
        params = {
            "function": "OVERVIEW",
            "symbol": symbol
        }
        
        data = self._make_request(params)
        
        if not data or "Symbol" not in data:
            raise ValueError(f"No data found for symbol: {symbol}")
        
        return data
    
    def get_global_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get current quote for a symbol
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            
        Returns:
            Dictionary with current quote information
        """
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol
        }
        
        data = self._make_request(params)
        
        quote = data.get("Global Quote", {})
        
        if not quote:
            raise ValueError(f"No quote found for symbol: {symbol}")
        
        # Convert numeric values
        for key, value in quote.items():
            if key not in ["01. symbol", "07. latest trading day"]:
                try:
                    quote[key] = float(value)
                except:
                    pass
        
        return quote 