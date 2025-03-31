"""
Market data service module.

This module provides services for fetching, processing, and analyzing market data.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
import os
import asyncio
from datetime import datetime, timedelta

# Import utility functions
from app.utils.market_data_utils import (
    fetch_stock_data,
    fetch_multiple_stocks,
    get_market_index_data,
    calculate_beta,
    save_stock_data,
    load_stock_data,
    check_data_freshness,
    analyze_stock_performance
)
from app.utils.data_utils import (
    process_stock_data,
    calculate_technical_indicators,
    normalize_data,
    calculate_returns,
    fix_yahoo_finance_data
)
from app.utils.file_utils import ensure_directory

# Import configuration
from app.config import (
    DATA_DIR,
    RESULTS_DIR,
    DEFAULT_PERIOD,
    DEFAULT_INTERVAL,
    CACHE_EXPIRY
)

# Set up logging
logger = logging.getLogger(__name__)

class MarketDataService:
    """Service for handling market data operations."""
    
    def __init__(self):
        """Initialize the market data service."""
        self.data_dir = DATA_DIR / "stocks"
        self.results_dir = RESULTS_DIR / "market_data"
        
        # Ensure directories exist
        ensure_directory(self.data_dir)
        ensure_directory(self.results_dir)
        
        # Track loaded data
        self._loaded_data = {}
        
        logger.info("Initialized MarketDataService")
    
    async def get_stock_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
        include_indicators: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get stock data for a ticker, either from cache or by fetching.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            period: Time period (e.g., '1d', '1mo', '1y')
            interval: Data interval (e.g., '1d', '1h', '15m')
            include_indicators: Whether to include technical indicators
            force_refresh: Whether to force refresh data from source
            
        Returns:
            DataFrame with stock data
        """
        # Check if data is already loaded in memory
        cache_key = f"{ticker}:{period}:{interval}"
        if not force_refresh and cache_key in self._loaded_data:
            logger.info(f"Using in-memory data for {ticker}")
            return self._loaded_data[cache_key]
        
        # Check if we should use cached file data
        needs_update = force_refresh or check_data_freshness(
            ticker=ticker,
            directory=self.data_dir,
            max_age_days=CACHE_EXPIRY // 24  # Convert hours to days
        )
        
        if needs_update:
            logger.info(f"Fetching fresh data for {ticker}")
            data = fetch_stock_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                period=period,
                interval=interval
            )
            
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Fix Yahoo Finance data issues
            data = fix_yahoo_finance_data(data)
            
            # Add technical indicators if requested
            if include_indicators:
                data = process_stock_data(data, add_indicators=True)
            
            # Save to file
            save_stock_data(data, ticker, self.data_dir)
        else:
            logger.info(f"Loading cached data for {ticker}")
            data = load_stock_data(ticker, self.data_dir)
            
            if data.empty:
                logger.warning(f"No cached data found for {ticker}, fetching fresh data")
                return await self.get_stock_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    period=period,
                    interval=interval,
                    include_indicators=include_indicators,
                    force_refresh=True
                )
            
            # Add technical indicators if requested and not already present
            if include_indicators and 'RSI' not in data.columns:
                data = process_stock_data(data, add_indicators=True)
        
        # Cache in memory
        self._loaded_data[cache_key] = data
        
        return data
    
    async def get_multiple_stocks(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = DEFAULT_PERIOD,
        interval: str = DEFAULT_INTERVAL,
        include_indicators: bool = True,
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date
            end_date: End date
            period: Time period
            interval: Data interval
            include_indicators: Whether to include technical indicators
            force_refresh: Whether to force refresh data from source
            
        Returns:
            Dictionary mapping tickers to DataFrames
        """
        results = {}
        
        # Process stocks concurrently
        async def process_stock(ticker):
            try:
                data = await self.get_stock_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    period=period,
                    interval=interval,
                    include_indicators=include_indicators,
                    force_refresh=force_refresh
                )
                if not data.empty:
                    results[ticker] = data
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
        
        # Process all stocks concurrently
        await asyncio.gather(
            *[process_stock(ticker) for ticker in tickers]
        )
        
        return results
    
    async def get_market_index(
        self,
        index_symbol: str = "^GSPC",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = DEFAULT_PERIOD,
        include_indicators: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get market index data.
        
        Args:
            index_symbol: Index ticker symbol
            start_date: Start date
            end_date: End date
            period: Time period
            include_indicators: Whether to include technical indicators
            force_refresh: Whether to force refresh data from source
            
        Returns:
            DataFrame with index data
        """
        # Common index mappings
        index_map = {
            "sp500": "^GSPC",
            "dow": "^DJI",
            "nasdaq": "^IXIC",
            "russell2000": "^RUT",
            "vix": "^VIX"
        }
        
        # Normalize index symbol
        symbol = index_map.get(index_symbol.lower(), index_symbol)
        
        # Get data using the same method as regular stocks
        return await self.get_stock_data(
            ticker=symbol,
            start_date=start_date,
            end_date=end_date,
            period=period,
            include_indicators=include_indicators,
            force_refresh=force_refresh
        )
    
    async def get_stock_beta(
        self,
        ticker: str,
        market_index: str = "^GSPC",
        period: int = 252,
        force_refresh: bool = False
    ) -> float:
        """
        Calculate beta for a stock.
        
        Args:
            ticker: Stock ticker symbol
            market_index: Market index symbol
            period: Number of days to use for calculation
            force_refresh: Whether to force refresh data from source
            
        Returns:
            Beta value
        """
        # Get stock data
        stock_data = await self.get_stock_data(
            ticker=ticker,
            period="1y" if period <= 252 else "2y",
            force_refresh=force_refresh
        )
        
        # Get market data
        market_data = await self.get_market_index(
            index_symbol=market_index,
            period="1y" if period <= 252 else "2y",
            force_refresh=force_refresh
        )
        
        # Calculate beta
        return calculate_beta(stock_data, market_data, period)
    
    async def analyze_stock(
        self,
        ticker: str,
        force_refresh: bool = False
    ) -> Dict[str, any]:
        """
        Perform comprehensive analysis on a stock.
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: Whether to force refresh data from source
            
        Returns:
            Dictionary with analysis results
        """
        # Get stock data
        stock_data = await self.get_stock_data(
            ticker=ticker,
            period="1y",
            include_indicators=True,
            force_refresh=force_refresh
        )
        
        if stock_data.empty:
            logger.warning(f"No data for {ticker}, cannot perform analysis")
            return {"error": "No data available for analysis"}
        
        # Perform performance analysis
        performance = analyze_stock_performance(stock_data)
        
        # Get market data and calculate beta
        try:
            beta = await self.get_stock_beta(ticker, force_refresh=force_refresh)
            performance["beta"] = beta
        except Exception as e:
            logger.error(f"Error calculating beta for {ticker}: {str(e)}")
            performance["beta"] = None
        
        # Get technical signals
        signals = self._get_technical_signals(stock_data)
        
        # Combine all analysis
        analysis = {
            "ticker": ticker,
            "performance": performance,
            "technical_signals": signals,
            "last_updated": datetime.now().isoformat()
        }
        
        return analysis
    
    def _get_technical_signals(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Generate technical analysis signals.
        
        Args:
            data: DataFrame with stock data including indicators
            
        Returns:
            Dictionary with technical signals
        """
        signals = {}
        
        # Check for required columns
        if data.empty or 'Close' not in data.columns:
            return {"error": "Insufficient data for technical analysis"}
        
        # Get latest data
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest
        
        # RSI signals
        if 'RSI' in data.columns:
            rsi = latest['RSI']
            signals['rsi'] = {
                'value': rsi,
                'signal': 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral',
                'strength': abs(50 - rsi) / 50  # 0 to 1
            }
        
        # MACD signals
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            macd = latest['MACD']
            signal = latest['MACD_Signal']
            prev_macd = prev['MACD']
            prev_signal = prev['MACD_Signal']
            
            # Check for crossovers
            current_diff = macd - signal
            prev_diff = prev_macd - prev_signal
            
            if current_diff > 0 and prev_diff <= 0:
                macd_signal = 'bullish_crossover'
            elif current_diff < 0 and prev_diff >= 0:
                macd_signal = 'bearish_crossover'
            elif current_diff > 0:
                macd_signal = 'bullish'
            elif current_diff < 0:
                macd_signal = 'bearish'
            else:
                macd_signal = 'neutral'
                
            signals['macd'] = {
                'value': macd,
                'signal_line': signal,
                'signal': macd_signal,
                'histogram': macd - signal
            }
        
        # Moving Average signals
        ma_columns = [col for col in data.columns if col.startswith('SMA_') or col.startswith('EMA_')]
        if ma_columns and 'Close' in data.columns:
            ma_signals = {}
            price = latest['Close']
            
            for ma_col in ma_columns:
                ma_value = latest[ma_col]
                
                # Determine if price is above or below MA
                if price > ma_value:
                    position = 'above'
                    strength = (price / ma_value - 1) * 100  # Percentage above
                else:
                    position = 'below'
                    strength = (ma_value / price - 1) * 100  # Percentage below
                
                ma_signals[ma_col] = {
                    'value': ma_value,
                    'position': position,
                    'strength': strength
                }
            
            signals['moving_averages'] = ma_signals
        
        # Bollinger Bands signals
        if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            upper = latest['BB_Upper']
            middle = latest['BB_Middle']
            lower = latest['BB_Lower']
            price = latest['Close']
            
            # Calculate width and position
            width = (upper - lower) / middle
            
            if price > upper:
                position = 'above_upper'
                deviation = (price - upper) / (upper - middle)
            elif price < lower:
                position = 'below_lower'
                deviation = (lower - price) / (middle - lower)
            else:
                position = 'within'
                # Calculate relative position within bands from -1 (at lower) to 1 (at upper)
                deviation = (2 * (price - lower) / (upper - lower)) - 1
                
            signals['bollinger_bands'] = {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'width': width,
                'position': position,
                'deviation': deviation
            }
            
        # Stochastic Oscillator signals
        if all(col in data.columns for col in ['%K', '%D']):
            k = latest['%K']
            d = latest['%D']
            
            if k < 20 and d < 20:
                stoch_signal = 'oversold'
            elif k > 80 and d > 80:
                stoch_signal = 'overbought'
            elif k > d and d < 50:
                stoch_signal = 'bullish_crossover'
            elif k < d and d > 50:
                stoch_signal = 'bearish_crossover'
            else:
                stoch_signal = 'neutral'
                
            signals['stochastic'] = {
                'k': k,
                'd': d,
                'signal': stoch_signal
            }
        
        # Overall signal
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        
        # Count bullish and bearish signals
        if 'rsi' in signals:
            total_signals += 1
            if signals['rsi']['signal'] == 'oversold':
                bullish_signals += 1
            elif signals['rsi']['signal'] == 'overbought':
                bearish_signals += 1
                
        if 'macd' in signals:
            total_signals += 1
            if 'bullish' in signals['macd']['signal']:
                bullish_signals += 1
            elif 'bearish' in signals['macd']['signal']:
                bearish_signals += 1
                
        if 'bollinger_bands' in signals:
            total_signals += 1
            if signals['bollinger_bands']['position'] == 'below_lower':
                bullish_signals += 1
            elif signals['bollinger_bands']['position'] == 'above_upper':
                bearish_signals += 1
                
        if 'stochastic' in signals:
            total_signals += 1
            if signals['stochastic']['signal'] == 'oversold':
                bullish_signals += 1
            elif signals['stochastic']['signal'] == 'overbought':
                bearish_signals += 1
        
        # Calculate signal strength
        if total_signals > 0:
            bullish_strength = bullish_signals / total_signals
            bearish_strength = bearish_signals / total_signals
            
            if bullish_strength > bearish_strength:
                overall_signal = 'bullish'
                strength = bullish_strength
            elif bearish_strength > bullish_strength:
                overall_signal = 'bearish'
                strength = bearish_strength
            else:
                overall_signal = 'neutral'
                strength = 0
                
            signals['overall'] = {
                'signal': overall_signal,
                'strength': strength,
                'bullish_count': bullish_signals,
                'bearish_count': bearish_signals,
                'total_indicators': total_signals
            }
        else:
            signals['overall'] = {
                'signal': 'insufficient_data',
                'strength': 0
            }
        
        return signals 