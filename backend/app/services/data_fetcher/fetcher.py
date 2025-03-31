"""
Data fetcher implementation for optimized financial data retrieval.

This module implements efficient data fetching with caching and rate limiting.
"""

import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Union
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import json
import aiohttp
import aiofiles
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import text

import yfinance as yf
from app.config import (
    API_KEYS, 
    CACHE_DIR, 
    CACHE_EXPIRY,
    DEFAULT_DATA_SOURCE, 
    DEFAULT_PERIOD, 
    DEFAULT_INTERVAL
)

logger = logging.getLogger(__name__)

# Rate limiting settings
RATE_LIMIT = {
    "yfinance": 0.2,  # seconds between requests
    "alpha_vantage": 0.5,  # seconds between requests
    "finnhub": 0.1  # seconds between requests
}

# Track the last request time for each API
last_request_time = {
    "yfinance": 0,
    "alpha_vantage": 0,
    "finnhub": 0
}

async def get_stock_data(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    indicators: bool = False,
    db: AsyncSession = None,
    source: str = DEFAULT_DATA_SOURCE
) -> Dict[str, Any]:
    """
    Get historical stock data for a symbol.
    
    Args:
        symbol: Stock symbol
        period: Time period (1d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        indicators: Whether to include technical indicators
        db: Database session
        source: Data source (yfinance, alpha_vantage)
        
    Returns:
        Dictionary containing stock data
    """
    # Check cache first
    cache_key = f"{symbol}_{period}_{interval}_{indicators}_{source}"
    cached_data = await check_cache(cache_key, "stock_data")
    
    if cached_data:
        logger.info(f"Using cached stock data for {symbol}")
        return cached_data
    
    # Apply rate limiting
    await apply_rate_limit(source)
    
    try:
        # Fetch data based on source
        if source == "yfinance":
            data = await fetch_from_yfinance(symbol, period, interval)
        elif source == "alpha_vantage":
            data = await fetch_from_alpha_vantage(symbol, period, interval)
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        # Calculate technical indicators if requested
        if indicators and not data.empty:
            data = await calculate_indicators(data)
        
        # Convert to dictionary for JSON serialization
        result = data_to_dict(data)
        
        # Cache the result
        await cache_data(cache_key, "stock_data", result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
        raise

async def get_stock_info(
    symbol: str,
    db: AsyncSession = None,
    source: str = DEFAULT_DATA_SOURCE
) -> Dict[str, Any]:
    """
    Get basic information about a stock.
    
    Args:
        symbol: Stock symbol
        db: Database session
        source: Data source (yfinance, alpha_vantage)
        
    Returns:
        Dictionary containing stock information
    """
    # Check cache first
    cache_key = f"{symbol}_info_{source}"
    cached_data = await check_cache(cache_key, "stock_info")
    
    if cached_data:
        logger.info(f"Using cached stock info for {symbol}")
        return cached_data
    
    # Apply rate limiting
    await apply_rate_limit(source)
    
    try:
        # Fetch data based on source
        if source == "yfinance":
            info = await fetch_info_from_yfinance(symbol)
        elif source == "alpha_vantage":
            info = await fetch_info_from_alpha_vantage(symbol)
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        # Cache the result
        await cache_data(cache_key, "stock_info", info)
        
        return info
        
    except Exception as e:
        logger.error(f"Error fetching stock info for {symbol}: {str(e)}")
        raise

async def search_stocks(
    query: str,
    limit: int = 10,
    db: AsyncSession = None
) -> List[Dict[str, Any]]:
    """
    Search for stocks by name or symbol.
    
    Args:
        query: Search query
        limit: Maximum number of results
        db: Database session
        
    Returns:
        List of matching stocks
    """
    # This would typically query a database or API
    # For now, we'll just return a mock result
    results = []
    
    # If we had a database, we would do something like:
    # async with db.begin():
    #     stmt = select(StockModel).where(
    #         StockModel.symbol.ilike(f"%{query}%") | 
    #         StockModel.name.ilike(f"%{query}%")
    #     ).limit(limit)
    #     
    #     result = await db.execute(stmt)
    #     stocks = result.scalars().all()
    #     
    #     for stock in stocks:
    #         results.append({
    #             "symbol": stock.symbol,
    #             "name": stock.name,
    #             "exchange": stock.exchange
    #         })
    
    return results

# Helper functions

async def apply_rate_limit(source: str) -> None:
    """
    Apply rate limiting for API requests.
    
    Args:
        source: Data source
    """
    if source in RATE_LIMIT:
        current_time = time.time()
        time_since_last_request = current_time - last_request_time.get(source, 0)
        
        if time_since_last_request < RATE_LIMIT[source]:
            # Wait to respect rate limit
            wait_time = RATE_LIMIT[source] - time_since_last_request
            logger.debug(f"Rate limiting {source} API, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        # Update last request time
        last_request_time[source] = time.time()

async def check_cache(key: str, cache_type: str) -> Optional[Dict[str, Any]]:
    """
    Check if data exists in cache and is not expired.
    
    Args:
        key: Cache key
        cache_type: Type of cached data
        
    Returns:
        Cached data if found and not expired, None otherwise
    """
    cache_dir = Path(CACHE_DIR) / cache_type
    cache_file = cache_dir / f"{key}.json"
    
    if not cache_file.exists():
        return None
    
    # Check if cache is expired
    file_stat = cache_file.stat()
    file_age = time.time() - file_stat.st_mtime
    
    if file_age > CACHE_EXPIRY * 3600:  # Convert hours to seconds
        logger.debug(f"Cache expired for {key}")
        return None
    
    try:
        async with aiofiles.open(cache_file, 'r') as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        logger.error(f"Error reading cache file {cache_file}: {str(e)}")
        return None

async def cache_data(key: str, cache_type: str, data: Dict[str, Any]) -> None:
    """
    Cache data to file.
    
    Args:
        key: Cache key
        cache_type: Type of data to cache
        data: Data to cache
    """
    cache_dir = Path(CACHE_DIR) / cache_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    
    try:
        async with aiofiles.open(cache_file, 'w') as f:
            await f.write(json.dumps(data, default=str))
            
        logger.debug(f"Cached data for {key}")
    except Exception as e:
        logger.error(f"Error caching data for {key}: {str(e)}")

def data_to_dict(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert DataFrame to dictionary for JSON serialization.
    
    Args:
        data: DataFrame to convert
        
    Returns:
        Dictionary representation of the DataFrame
    """
    if data.empty:
        return {}
    
    # Reset index to make date a column
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index()
    
    # Convert dates to strings
    if 'Date' in data.columns:
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    elif 'date' in data.columns:
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
    
    # Handle NaN values
    data = data.replace({np.nan: None})
    
    # Convert to dictionary
    records = data.to_dict(orient='records')
    columns = list(data.columns)
    
    return {
        "columns": columns,
        "data": records
    }

async def fetch_from_yfinance(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        symbol: Stock symbol
        period: Time period
        interval: Data interval
        
    Returns:
        DataFrame containing stock data
    """
    logger.info(f"Fetching data from Yahoo Finance for {symbol}, period={period}, interval={interval}")
    
    # Since yfinance is synchronous, run it in a thread pool
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(
        None, 
        lambda: yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    )
    
    # Rename columns to lowercase
    data.columns = [col.lower() for col in data.columns]
    
    return data

async def fetch_from_alpha_vantage(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch stock data from Alpha Vantage API.
    
    Args:
        symbol: Stock symbol
        period: Time period
        interval: Data interval
        
    Returns:
        DataFrame containing stock data
    """
    logger.info(f"Fetching data from Alpha Vantage for {symbol}")
    
    # Map period and interval to Alpha Vantage parameters
    av_interval = map_interval_to_alpha_vantage(interval)
    
    # Build API URL
    api_key = API_KEYS.get("alpha_vantage", "")
    if not api_key:
        raise ValueError("Alpha Vantage API key not found")
    
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={av_interval}&outputsize=full&apikey={api_key}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Error fetching data from Alpha Vantage: {response.status}")
                raise ValueError(f"Failed to fetch data: HTTP {response.status}")
            
            result = await response.json()
            
            # Parse the response
            time_series_key = f"Time Series ({av_interval})"
            if time_series_key not in result:
                logger.error(f"Unexpected response from Alpha Vantage: {result}")
                raise ValueError("Unexpected response format from Alpha Vantage")
            
            time_series = result[time_series_key]
            
            # Convert to DataFrame
            data = pd.DataFrame.from_dict(time_series, orient='index')
            
            # Rename columns
            data.columns = [col.split(' ')[1].lower() for col in data.columns]
            
            # Convert index to datetime
            data.index = pd.to_datetime(data.index)
            
            # Sort by date
            data = data.sort_index()
            
            # Only return the requested period
            if period != "max":
                days = convert_period_to_days(period)
                start_date = pd.Timestamp.now() - pd.Timedelta(days=days)
                data = data[data.index >= start_date]
            
            return data

async def fetch_info_from_yfinance(symbol: str) -> Dict[str, Any]:
    """
    Fetch stock information from Yahoo Finance.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary containing stock information
    """
    logger.info(f"Fetching stock info from Yahoo Finance for {symbol}")
    
    # Since yfinance is synchronous, run it in a thread pool
    loop = asyncio.get_event_loop()
    ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
    
    # Get the info dictionary
    info = await loop.run_in_executor(None, lambda: ticker.info)
    
    # Clean up the info dictionary
    # Convert data types to JSON-serializable types
    cleaned_info = {}
    for key, value in info.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            cleaned_info[key] = value
        else:
            # Convert non-standard types to string
            cleaned_info[key] = str(value)
    
    return cleaned_info

async def fetch_info_from_alpha_vantage(symbol: str) -> Dict[str, Any]:
    """
    Fetch stock information from Alpha Vantage.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary containing stock information
    """
    logger.info(f"Fetching stock info from Alpha Vantage for {symbol}")
    
    # Build API URL
    api_key = API_KEYS.get("alpha_vantage", "")
    if not api_key:
        raise ValueError("Alpha Vantage API key not found")
    
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={api_key}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Error fetching info from Alpha Vantage: {response.status}")
                raise ValueError(f"Failed to fetch info: HTTP {response.status}")
            
            result = await response.json()
            
            # Check if the response is valid
            if "Symbol" not in result:
                logger.error(f"Unexpected response from Alpha Vantage: {result}")
                raise ValueError("Unexpected response format from Alpha Vantage")
            
            return result

async def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for stock data.
    
    Args:
        data: DataFrame containing stock data
        
    Returns:
        DataFrame with added technical indicators
    """
    # Copy the DataFrame to avoid modifying the original
    df = data.copy()
    
    # Calculate SMA
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Calculate EMA
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # Calculate MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * std
    df['bb_lower'] = df['bb_middle'] - 2 * std
    
    return df

def map_interval_to_alpha_vantage(interval: str) -> str:
    """
    Map Yahoo Finance interval to Alpha Vantage interval.
    
    Args:
        interval: Yahoo Finance interval
        
    Returns:
        Alpha Vantage interval
    """
    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "60min",
        # Alpha Vantage doesn't support daily intervals in the intraday API,
        # but we'll default to the highest resolution they offer for other intervals
        "1d": "60min",
        "1wk": "60min",
        "1mo": "60min"
    }
    
    return mapping.get(interval, "60min")

def convert_period_to_days(period: str) -> int:
    """
    Convert Yahoo Finance period to number of days.
    
    Args:
        period: Yahoo Finance period
        
    Returns:
        Number of days
    """
    mapping = {
        "1d": 1,
        "5d": 5,
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
        "10y": 3650,
        "ytd": (datetime.datetime.now() - datetime.datetime(datetime.datetime.now().year, 1, 1)).days
    }
    
    return mapping.get(period, 365)  # Default to 1 year 