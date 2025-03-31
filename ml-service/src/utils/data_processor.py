#!/usr/bin/env python3
"""
Data Processor for Investment Recommendation System
Handles data fetching, caching, and preprocessing for ML models
"""

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import logging
import functools
import time
from typing import Dict, List, Union, Optional, Tuple

# Technical indicator imports
import ta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, cache_duration: int = 15):
        """Initialize data processor with cache duration in minutes"""
        self.data_cache = {}
        self.cache_duration = cache_duration * 60  # Convert to seconds
    
    def get_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data for a symbol with caching
        Periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        Intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        """
        cache_key = f"{symbol}_{period}_{interval}"
        current_time = time.time()
        
        # Check if data is in cache and not expired
        if (
            cache_key in self.data_cache and
            current_time - self.data_cache[cache_key]["timestamp"] < self.cache_duration
        ):
            logger.info(f"Returning cached data for {symbol}")
            return self.data_cache[cache_key]["data"]
        
        try:
            logger.info(f"Fetching data for {symbol}")
            data = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            # Verify data is valid
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Clean up data
            data = data.copy()
            data.index = pd.to_datetime(data.index)
            data = data.dropna()
            
            # Cache the data
            self.data_cache[cache_key] = {
                "data": data,
                "timestamp": current_time
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            
            # Return empty dataframe on error
            return pd.DataFrame()
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to price data"""
        if data.empty:
            return data
            
        df = data.copy()
        
        try:
            # Initialize feature columns
            df = df.assign(Date=df.index)
            
            # Add trend indicators (Moving averages, MACD)
            df["SMA20"] = ta.trend.sma_indicator(df["Close"], window=20)
            df["SMA50"] = ta.trend.sma_indicator(df["Close"], window=50)
            df["SMA200"] = ta.trend.sma_indicator(df["Close"], window=200)
            
            macd = ta.trend.MACD(df["Close"])
            df["MACD"] = macd.macd()
            df["MACD_Signal"] = macd.macd_signal()
            df["MACD_Diff"] = macd.macd_diff()
            
            # Add momentum indicators (RSI, Stochastic)
            df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
            
            stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
            df["Stoch_K"] = stoch.stoch()
            df["Stoch_D"] = stoch.stoch_signal()
            
            # Add volatility indicators (Bollinger Bands, ATR)
            bollinger = ta.volatility.BollingerBands(df["Close"])
            df["BB_High"] = bollinger.bollinger_hband()
            df["BB_Low"] = bollinger.bollinger_lband()
            df["BB_Width"] = bollinger.bollinger_wband()
            
            df["ATR"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
            
            # Add volume indicators
            df["Volume_SMA20"] = ta.volume.sma_volume(df["Volume"], window=20)
            
            # Add price to moving average ratios
            df["Price_to_SMA20"] = df["Close"] / df["SMA20"]
            df["Price_to_SMA50"] = df["Close"] / df["SMA50"]
            
            # Clean up NaN values
            df = df.fillna(method="bfill")
            df = df.fillna(0)
            
            # Reset index to maintain date as index
            df = df.set_index("Date")
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def get_market_data(self) -> Dict[str, pd.DataFrame]:
        """Get overall market data for major indices"""
        indices = [
            "^GSPC",  # S&P 500
            "^DJI",   # Dow Jones
            "^IXIC",  # NASDAQ
            "^RUT"    # Russell 2000
        ]
        
        market_data = {}
        for index in indices:
            market_data[index] = self.get_stock_data(index, period="1y")
            
        return market_data
    
    def get_sector_performance(self, period: str = "1mo") -> pd.DataFrame:
        """Get performance data for market sectors"""
        sectors = [
            "XLK",  # Technology
            "XLF",  # Financials
            "XLV",  # Healthcare
            "XLE",  # Energy
            "XLI",  # Industrials
            "XLP",  # Consumer Staples
            "XLY",  # Consumer Discretionary
            "XLB",  # Materials
            "XLU",  # Utilities
            "XLRE"  # Real Estate
        ]
        
        sector_data = {}
        for sector in sectors:
            data = self.get_stock_data(sector, period=period)
            if not data.empty:
                start_price = data.iloc[0]["Close"]
                current_price = data.iloc[-1]["Close"]
                sector_data[sector] = {
                    "performance": (current_price - start_price) / start_price * 100,
                    "current_price": current_price,
                    "start_price": start_price,
                    "volatility": data["Close"].pct_change().std() * 100
                }
                
        return pd.DataFrame(sector_data).T
    
    def get_correlation_matrix(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Calculate correlation matrix between symbols"""
        price_data = {}
        
        for symbol in symbols:
            data = self.get_stock_data(symbol, period=period)
            if not data.empty:
                price_data[symbol] = data["Close"]
                
        if not price_data:
            return pd.DataFrame()
            
        # Create DataFrame with all price series
        df = pd.DataFrame(price_data)
        
        # Calculate correlation matrix
        return df.pct_change().corr()
    
    def get_risk_metrics(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Calculate risk metrics for a list of symbols"""
        risk_data = {}
        
        for symbol in symbols:
            data = self.get_stock_data(symbol, period=period)
            if not data.empty:
                returns = data["Close"].pct_change().dropna()
                
                risk_data[symbol] = {
                    "volatility": returns.std() * np.sqrt(252) * 100,  # Annualized
                    "sharpe": (returns.mean() / returns.std()) * np.sqrt(252),
                    "max_drawdown": self._calculate_max_drawdown(data["Close"]) * 100,
                    "var_95": returns.quantile(0.05) * 100,  # 95% VaR
                    "mean_return": returns.mean() * 100
                }
                
        return pd.DataFrame(risk_data).T
    
    def _calculate_max_drawdown(self, price_series: pd.Series) -> float:
        """Calculate maximum drawdown for a price series"""
        cumulative_max = price_series.cummax()
        drawdown = (price_series - cumulative_max) / cumulative_max
        return drawdown.min()
    
    def prepare_lstm_input(
        self,
        symbol: str,
        lookback_period: int = 60,
        use_tech_indicators: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Prepare input data for LSTM model"""
        # Get data
        data = self.get_stock_data(symbol, period="2y")
        
        if use_tech_indicators:
            data = self.add_technical_indicators(data)
        
        # Select features (exclude redundant columns)
        exclude_cols = ["Open", "High", "Low", "Volume", "Adj Close"]
        features = [col for col in data.columns if col not in exclude_cols]
        
        # Scale data using Min-Max scaling
        scaler = None  # This would be a MinMaxScaler in actual implementation
        
        # Create sequences
        X, y = [], []
        for i in range(lookback_period, len(data)):
            X.append(data[features].values[i-lookback_period:i])
            y.append(data["Close"].values[i])
        
        return np.array(X), np.array(y), data 