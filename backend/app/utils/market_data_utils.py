"""
Market data utility functions for the investment recommendation system.

This module provides utility functions for fetching, processing, and analyzing
market data from various sources.
"""
import os
import pandas as pd
import numpy as np
import logging
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def fetch_stock_data(ticker: str, start_date: Optional[str] = None, 
                    end_date: Optional[str] = None,
                    period: Optional[str] = None,
                    interval: str = "1d") -> pd.DataFrame:
    """
    Fetch stock data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        period: Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
                If provided, start_date and end_date are ignored
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
    Returns:
        DataFrame with stock data
    """
    try:
        # Validate inputs
        if not period and not (start_date and end_date):
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            logger.info(f"Using default date range: {start_date} to {end_date}")
            
        # Fetch data
        ticker_obj = yf.Ticker(ticker)
        
        if period:
            data = ticker_obj.history(period=period, interval=interval)
        else:
            data = ticker_obj.history(start=start_date, end=end_date, interval=interval)
        
        # Check if data is empty
        if data.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
            
        # Clean up data
        data = data.dropna(how='all')
        
        # Convert column names to standard format
        data.columns = [col.title() for col in data.columns]
        
        # Check for Adj Close and rename
        if 'Adj Close' in data.columns:
            data.rename(columns={'Adj Close': 'AdjClose'}, inplace=True)
        
        logger.info(f"Successfully fetched {len(data)} records for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
        return pd.DataFrame()

def fetch_multiple_stocks(tickers: List[str], start_date: Optional[str] = None,
                         end_date: Optional[str] = None, 
                         period: Optional[str] = None,
                         interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple stock tickers.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date
        end_date: End date
        period: Time period
        interval: Data interval
        
    Returns:
        Dictionary mapping tickers to DataFrames
    """
    results = {}
    
    for ticker in tickers:
        try:
            data = fetch_stock_data(ticker, start_date, end_date, period, interval)
            if not data.empty:
                results[ticker] = data
            else:
                logger.warning(f"Skipping {ticker}, no data returned")
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
    
    logger.info(f"Successfully fetched data for {len(results)}/{len(tickers)} stocks")
    return results

def get_market_index_data(index_symbol: str = "^GSPC", 
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         period: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch market index data (e.g., S&P 500, NASDAQ, etc.)
    
    Args:
        index_symbol: Index ticker symbol
        start_date: Start date
        end_date: End date
        period: Time period
        
    Returns:
        DataFrame with index data
    """
    # Common indices:
    # ^GSPC - S&P 500
    # ^DJI - Dow Jones Industrial Average
    # ^IXIC - NASDAQ Composite
    # ^RUT - Russell 2000
    # ^VIX - CBOE Volatility Index
    
    index_map = {
        "SP500": "^GSPC",
        "DOW": "^DJI",
        "NASDAQ": "^IXIC",
        "RUSSELL2000": "^RUT",
        "VIX": "^VIX"
    }
    
    # If a common name was provided, convert to symbol
    symbol = index_map.get(index_symbol.upper(), index_symbol)
    
    try:
        data = fetch_stock_data(symbol, start_date, end_date, period)
        if not data.empty:
            logger.info(f"Successfully fetched {index_symbol} market index data")
        return data
    except Exception as e:
        logger.error(f"Error fetching market index data for {index_symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_beta(stock_data: pd.DataFrame, market_data: pd.DataFrame, 
                  period: int = 252) -> float:
    """
    Calculate beta (market correlation/risk) for a stock.
    
    Args:
        stock_data: DataFrame with stock price data
        market_data: DataFrame with market index data
        period: Number of days to use for calculation
        
    Returns:
        Beta value
    """
    try:
        # Ensure both have the same dates
        common_dates = stock_data.index.intersection(market_data.index)
        if len(common_dates) < 30:  # Need sufficient data
            logger.warning(f"Insufficient data to calculate beta: {len(common_dates)} common dates")
            return np.nan
            
        # Get returns
        stock_returns = stock_data.loc[common_dates, 'Close'].pct_change().dropna()
        market_returns = market_data.loc[common_dates, 'Close'].pct_change().dropna()
        
        # Align data
        aligned_data = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        }).dropna()
        
        # Take last N periods
        if len(aligned_data) > period:
            aligned_data = aligned_data.iloc[-period:]
            
        # Calculate beta: covariance(stock, market) / variance(market)
        covariance = aligned_data['stock'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()
        
        beta = covariance / market_variance
        logger.info(f"Calculated beta: {beta:.4f}")
        return beta
        
    except Exception as e:
        logger.error(f"Error calculating beta: {str(e)}")
        return np.nan

def save_stock_data(data: pd.DataFrame, ticker: str, directory: Union[str, Path] = "data/stocks") -> bool:
    """
    Save stock data to file.
    
    Args:
        data: DataFrame with stock data
        ticker: Stock ticker symbol
        directory: Directory to save data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save file
        filepath = dir_path / f"{ticker.replace('.', '_')}.csv"
        data.to_csv(filepath)
        
        # Save metadata
        metadata = {
            "ticker": ticker,
            "start_date": data.index.min().strftime("%Y-%m-%d"),
            "end_date": data.index.max().strftime("%Y-%m-%d"),
            "rows": len(data),
            "columns": list(data.columns),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = dir_path / f"{ticker.replace('.', '_')}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved {ticker} data to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving stock data for {ticker}: {str(e)}")
        return False

def load_stock_data(ticker: str, directory: Union[str, Path] = "data/stocks") -> pd.DataFrame:
    """
    Load stock data from file.
    
    Args:
        ticker: Stock ticker symbol
        directory: Directory where data is stored
        
    Returns:
        DataFrame with stock data
    """
    try:
        # Construct filepath
        dir_path = Path(directory)
        filepath = dir_path / f"{ticker.replace('.', '_')}.csv"
        
        # Check if file exists
        if not filepath.exists():
            logger.warning(f"Stock data file for {ticker} not found at {filepath}")
            return pd.DataFrame()
            
        # Load data
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded {ticker} data from {filepath}: {len(data)} rows")
        return data
        
    except Exception as e:
        logger.error(f"Error loading stock data for {ticker}: {str(e)}")
        return pd.DataFrame()

def check_data_freshness(ticker: str, directory: Union[str, Path] = "data/stocks", 
                        max_age_days: int = 1) -> bool:
    """
    Check if stock data is recent enough or needs to be updated.
    
    Args:
        ticker: Stock ticker symbol
        directory: Directory where data is stored
        max_age_days: Maximum allowed age of data in days
        
    Returns:
        True if data needs to be updated, False otherwise
    """
    try:
        # Construct metadata filepath
        dir_path = Path(directory)
        metadata_path = dir_path / f"{ticker.replace('.', '_')}_metadata.json"
        
        # Check if metadata file exists
        if not metadata_path.exists():
            logger.info(f"No metadata found for {ticker}, needs update")
            return True
            
        # Load metadata
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Check last updated date
        last_updated = datetime.strptime(metadata["last_updated"], "%Y-%m-%d %H:%M:%S")
        age = datetime.now() - last_updated
        
        if age.days > max_age_days:
            logger.info(f"Data for {ticker} is {age.days} days old, needs update")
            return True
            
        logger.info(f"Data for {ticker} is recent ({age.days} days old)")
        return False
        
    except Exception as e:
        logger.error(f"Error checking data freshness for {ticker}: {str(e)}")
        # If error occurs, assume data needs update
        return True

def analyze_stock_performance(data: pd.DataFrame, periods: List[int] = [7, 30, 90, 365]) -> Dict[str, float]:
    """
    Analyze stock performance over different time periods.
    
    Args:
        data: DataFrame with stock price data
        periods: List of periods (in days) to analyze
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        # Ensure data is sorted by date
        data = data.sort_index()
        
        # Initialize results
        results = {}
        
        # Calculate returns for different periods
        for period in periods:
            if len(data) >= period:
                # Recent data subset
                recent = data.iloc[-period:]
                
                # Return
                start_price = recent['Close'].iloc[0]
                end_price = recent['Close'].iloc[-1]
                period_return = (end_price / start_price - 1) * 100
                results[f"{period}d_return"] = period_return
                
                # Volatility (annualized)
                daily_returns = recent['Close'].pct_change().dropna()
                volatility = daily_returns.std() * (252 ** 0.5) * 100  # Annualized
                results[f"{period}d_volatility"] = volatility
                
                # Max drawdown
                cumulative = (1 + daily_returns).cumprod()
                max_dd = (cumulative / cumulative.cummax() - 1).min() * 100
                results[f"{period}d_max_drawdown"] = max_dd
                
                # Risk-adjusted return (Sharpe ratio approximation)
                risk_free_rate = 0.03  # Assumption
                excess_return = period_return - (risk_free_rate / 365 * period)
                sharpe = excess_return / volatility if volatility > 0 else 0
                results[f"{period}d_sharpe"] = sharpe
        
        # Add overall metrics
        if len(data) > 0:
            # Current price
            results["current_price"] = data['Close'].iloc[-1]
            
            # 52-week high/low
            year_data = data.iloc[-252:] if len(data) >= 252 else data
            results["52w_high"] = year_data['High'].max()
            results["52w_low"] = year_data['Low'].min()
            results["52w_high_pct"] = (results["current_price"] / results["52w_high"] - 1) * 100
            results["52w_low_pct"] = (results["current_price"] / results["52w_low"] - 1) * 100
            
            # Average volume
            if 'Volume' in data.columns:
                results["avg_volume"] = data['Volume'].mean()
                results["recent_volume"] = data['Volume'].iloc[-5:].mean()
                results["volume_ratio"] = results["recent_volume"] / results["avg_volume"] if results["avg_volume"] > 0 else 0
        
        logger.info(f"Analyzed stock performance over {len(periods)} time periods")
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing stock performance: {str(e)}")
        return {} 