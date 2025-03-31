"""
Data utility functions for the investment recommendation system.

This module provides utility functions for data processing, transformation,
and analysis related operations.
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for stock price data.
    
    Args:
        df: DataFrame with OHLC price data
        
    Returns:
        DataFrame with added technical indicators
    """
    try:
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in result.columns for col in required_columns):
            logger.error(f"DataFrame missing required columns: {required_columns}")
            return df
        
        # Moving Averages
        result['SMA_5'] = result['Close'].rolling(window=5).mean()
        result['SMA_10'] = result['Close'].rolling(window=10).mean()
        result['SMA_20'] = result['Close'].rolling(window=20).mean()
        result['SMA_50'] = result['Close'].rolling(window=50).mean()
        result['SMA_100'] = result['Close'].rolling(window=100).mean()
        result['SMA_200'] = result['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        result['EMA_5'] = result['Close'].ewm(span=5, adjust=False).mean()
        result['EMA_10'] = result['Close'].ewm(span=10, adjust=False).mean()
        result['EMA_20'] = result['Close'].ewm(span=20, adjust=False).mean()
        result['EMA_50'] = result['Close'].ewm(span=50, adjust=False).mean()
        result['EMA_100'] = result['Close'].ewm(span=100, adjust=False).mean()
        result['EMA_200'] = result['Close'].ewm(span=200, adjust=False).mean()
        
        # Bollinger Bands (20-day, 2 standard deviations)
        result['BB_Middle'] = result['Close'].rolling(window=20).mean()
        result['BB_StdDev'] = result['Close'].rolling(window=20).std()
        result['BB_Upper'] = result['BB_Middle'] + (result['BB_StdDev'] * 2)
        result['BB_Lower'] = result['BB_Middle'] - (result['BB_StdDev'] * 2)
        
        # MACD (Moving Average Convergence Divergence)
        result['EMA_12'] = result['Close'].ewm(span=12, adjust=False).mean()
        result['EMA_26'] = result['Close'].ewm(span=26, adjust=False).mean()
        result['MACD'] = result['EMA_12'] - result['EMA_26']
        result['MACD_Signal'] = result['MACD'].ewm(span=9, adjust=False).mean()
        result['MACD_Hist'] = result['MACD'] - result['MACD_Signal']
        
        # RSI (Relative Strength Index, 14 periods)
        delta = result['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        result['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator (14,3,3)
        result['14-high'] = result['High'].rolling(14).max()
        result['14-low'] = result['Low'].rolling(14).min()
        result['%K'] = (result['Close'] - result['14-low']) * 100 / (result['14-high'] - result['14-low'])
        result['%D'] = result['%K'].rolling(3).mean()
        
        # Average True Range (ATR, 14 periods)
        tr1 = result['High'] - result['Low']
        tr2 = abs(result['High'] - result['Close'].shift())
        tr3 = abs(result['Low'] - result['Close'].shift())
        result['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result['ATR'] = result['TR'].rolling(14).mean()
        
        # Rate of Change (ROC, 10 periods)
        result['ROC'] = ((result['Close'] / result['Close'].shift(10)) - 1) * 100
        
        # On-Balance Volume (OBV) - if Volume column exists
        if 'Volume' in result.columns:
            obv = pd.Series(index=result.index)
            obv.iloc[0] = 0
            for i in range(1, len(result)):
                if result['Close'].iloc[i] > result['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + result['Volume'].iloc[i]
                elif result['Close'].iloc[i] < result['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - result['Volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            result['OBV'] = obv
        
        # Remove temporary columns
        if '14-high' in result.columns:
            result = result.drop(['14-high', '14-low'], axis=1)
        
        # Fill NaN values at the beginning
        result = result.replace([np.inf, -np.inf], np.nan)
        
        logger.info("Successfully calculated technical indicators")
        return result
    
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {str(e)}")
        return df

def resample_timeseries(df: pd.DataFrame, period: str = 'D') -> pd.DataFrame:
    """
    Resample time series data to a specified frequency.
    
    Args:
        df: DataFrame with time series data (index must be DatetimeIndex)
        period: Resampling period ('D' for daily, 'W' for weekly, 'M' for monthly, etc.)
        
    Returns:
        Resampled DataFrame
    """
    try:
        # Ensure the index is a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, attempting to convert")
            if 'Date' in df.columns:
                df = df.set_index('Date')
            df.index = pd.to_datetime(df.index)
        
        # Define how to resample each column
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Create a dictionary for all columns
        resample_dict = {}
        for col in df.columns:
            if col in ohlc_dict:
                resample_dict[col] = ohlc_dict[col]
            else:
                # For other columns, use the last value
                resample_dict[col] = 'last'
        
        # Resample
        resampled = df.resample(period).agg(resample_dict)
        
        logger.info(f"Successfully resampled data to {period} frequency")
        return resampled
    
    except Exception as e:
        logger.error(f"Error resampling time series data: {str(e)}")
        return df

def normalize_data(df: pd.DataFrame, method: str = 'minmax', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize numerical columns in the DataFrame.
    
    Args:
        df: DataFrame to normalize
        method: Normalization method ('minmax', 'zscore', 'robust')
        columns: List of columns to normalize (all numerical columns if None)
        
    Returns:
        DataFrame with normalized columns
    """
    try:
        result = df.copy()
        
        # Determine columns to normalize
        if columns is None:
            columns = result.select_dtypes(include=['number']).columns.tolist()
        
        for col in columns:
            if col not in result.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            if method == 'minmax':
                # Min-Max Normalization (scale to 0-1)
                min_val = result[col].min()
                max_val = result[col].max()
                result[col] = (result[col] - min_val) / (max_val - min_val)
            
            elif method == 'zscore':
                # Z-score Normalization (mean 0, std 1)
                mean = result[col].mean()
                std = result[col].std()
                result[col] = (result[col] - mean) / std
            
            elif method == 'robust':
                # Robust Scaling (based on median and IQR)
                median = result[col].median()
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                result[col] = (result[col] - median) / iqr
            
            else:
                logger.warning(f"Unsupported normalization method: {method}")
        
        logger.info(f"Successfully normalized {len(columns)} columns using {method} method")
        return result
    
    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
        return df

def calculate_returns(df: pd.DataFrame, column: str = 'Close', periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Calculate returns over specified periods.
    
    Args:
        df: DataFrame with price data
        column: Column to calculate returns from
        periods: List of periods to calculate returns for
        
    Returns:
        DataFrame with added return columns
    """
    try:
        result = df.copy()
        
        # Check if column exists
        if column not in result.columns:
            logger.error(f"Column {column} not found in DataFrame")
            return df
        
        # Calculate returns for each period
        for period in periods:
            result[f'Return_{period}d'] = result[column].pct_change(period)
        
        logger.info(f"Successfully calculated returns for periods: {periods}")
        return result
    
    except Exception as e:
        logger.error(f"Error calculating returns: {str(e)}")
        return df

def split_time_series(df: pd.DataFrame, train_size: float = 0.8, test_size: Optional[float] = None, 
                     val_size: Optional[float] = None) -> Dict[str, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets.
    
    Args:
        df: DataFrame with time series data
        train_size: Proportion of data for training set
        test_size: Proportion of data for test set (calculated from remainder if None)
        val_size: Proportion of data for validation set (0 if None)
        
    Returns:
        Dictionary with train, validation, and test DataFrames
    """
    try:
        # Calculate proportions if not provided
        if test_size is None and val_size is None:
            test_size = 1 - train_size
            val_size = 0
        elif test_size is None:
            test_size = 1 - train_size - val_size
        elif val_size is None:
            val_size = 1 - train_size - test_size
        
        # Ensure proportions sum to 1
        if not np.isclose(train_size + test_size + val_size, 1.0):
            logger.warning(f"Proportions do not sum to 1: {train_size + test_size + val_size}")
            # Normalize proportions
            total = train_size + test_size + val_size
            train_size /= total
            test_size /= total
            val_size /= total
        
        # Calculate indices
        n = len(df)
        train_end = int(n * train_size)
        val_end = train_end + int(n * val_size)
        
        # Split data
        result = {
            'train': df.iloc[:train_end].copy(),
            'val': df.iloc[train_end:val_end].copy() if val_size > 0 else None,
            'test': df.iloc[val_end:].copy()
        }
        
        # Log info about splits
        logger.info(f"Split data: train={len(result['train'])} rows, "
                  f"val={len(result['val']) if result['val'] is not None else 0} rows, "
                  f"test={len(result['test'])} rows")
        
        return result
    
    except Exception as e:
        logger.error(f"Error splitting time series data: {str(e)}")
        return {'train': df, 'val': None, 'test': None}

def detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'zscore', 
                   threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect outliers in numerical columns.
    
    Args:
        df: DataFrame to check for outliers
        columns: List of columns to check (all numerical columns if None)
        method: Detection method ('zscore', 'iqr')
        threshold: Threshold for detection (zscore: typically 3.0, iqr: typically 1.5)
        
    Returns:
        DataFrame with 'is_outlier' column
    """
    try:
        result = df.copy()
        
        # Add outlier flag column
        result['is_outlier'] = False
        
        # Determine columns to check
        if columns is None:
            columns = result.select_dtypes(include=['number']).columns.tolist()
        
        for col in columns:
            if col not in result.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            if method == 'zscore':
                # Z-score method
                mean = result[col].mean()
                std = result[col].std()
                z_scores = np.abs((result[col] - mean) / std)
                result.loc[z_scores > threshold, 'is_outlier'] = True
            
            elif method == 'iqr':
                # IQR method
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (threshold * iqr)
                upper_bound = q3 + (threshold * iqr)
                result.loc[(result[col] < lower_bound) | (result[col] > upper_bound), 'is_outlier'] = True
            
            else:
                logger.warning(f"Unsupported outlier detection method: {method}")
        
        outlier_count = result['is_outlier'].sum()
        logger.info(f"Detected {outlier_count} outliers using {method} method with threshold {threshold}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        return df

def process_stock_data(df: pd.DataFrame, add_indicators: bool = True, 
                      resample_period: Optional[str] = None,
                      calculate_ret: bool = True,
                      fill_na: bool = True) -> pd.DataFrame:
    """
    Process stock data with common operations in one function.
    
    Args:
        df: DataFrame with OHLC price data
        add_indicators: Whether to add technical indicators
        resample_period: Period to resample data to (None for no resampling)
        calculate_ret: Whether to calculate returns
        fill_na: Whether to fill NaN values
        
    Returns:
        Processed DataFrame
    """
    try:
        result = df.copy()
        
        # Convert date index if needed
        if not isinstance(result.index, pd.DatetimeIndex):
            if 'Date' in result.columns:
                result = result.set_index('Date')
            result.index = pd.to_datetime(result.index)
        
        # Ensure column names are standardized
        result.columns = [col.title() if col.lower() in ['open', 'high', 'low', 'close', 'volume'] 
                         else col for col in result.columns]
        
        # Resample if needed
        if resample_period:
            result = resample_timeseries(result, period=resample_period)
        
        # Add technical indicators
        if add_indicators:
            result = calculate_technical_indicators(result)
        
        # Calculate returns
        if calculate_ret:
            result = calculate_returns(result)
        
        # Fill NaN values
        if fill_na:
            result = result.replace([np.inf, -np.inf], np.nan)
            # Forward fill first, then backward fill remaining NaNs
            result = result.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Processed stock data: {len(result)} rows, {len(result.columns)} columns")
        return result
    
    except Exception as e:
        logger.error(f"Error processing stock data: {str(e)}")
        return df

def prepare_data_for_forecasting(df: pd.DataFrame, 
                               target_col: str = 'Close',
                               feature_cols: Optional[List[str]] = None,
                               lag_periods: List[int] = [1, 2, 3, 5, 10],
                               target_shifts: List[int] = [1, 3, 5]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for forecasting models by creating lagged features and future targets.
    
    Args:
        df: DataFrame with time series data
        target_col: Column to predict
        feature_cols: Columns to use as features (None for all numerical columns)
        lag_periods: List of periods to lag for features
        target_shifts: List of future periods to predict
        
    Returns:
        Tuple of (features DataFrame, targets DataFrame)
    """
    try:
        # Make a copy
        data = df.copy()
        
        # Determine feature columns if not specified
        if feature_cols is None:
            feature_cols = [col for col in data.select_dtypes(include=['number']).columns 
                          if col != target_col]
        
        # Create lagged features
        for col in feature_cols:
            if col not in data.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
                
            for lag in lag_periods:
                data[f"{col}_lag_{lag}"] = data[col].shift(lag)
        
        # Create future targets
        targets = pd.DataFrame(index=data.index)
        for shift in target_shifts:
            targets[f"{target_col}_future_{shift}"] = data[target_col].shift(-shift)
        
        # Remove rows with NaN values (caused by shifting)
        valid_idx = ~data.isnull().any(axis=1) & ~targets.isnull().any(axis=1)
        features = data.loc[valid_idx]
        targets = targets.loc[valid_idx]
        
        logger.info(f"Prepared data for forecasting: {len(features)} samples, " 
                  f"{len(features.columns)} features, {len(targets.columns)} targets")
        
        return features, targets
    
    except Exception as e:
        logger.error(f"Error preparing data for forecasting: {str(e)}")
        # Return empty DataFrames to indicate error
        return pd.DataFrame(), pd.DataFrame()

def fix_yahoo_finance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix common issues with data from Yahoo Finance.
    
    Args:
        df: DataFrame from yfinance
        
    Returns:
        Fixed DataFrame
    """
    try:
        result = df.copy()
        
        # Standard column names
        if result.columns.str.contains('Adj Close').any():
            result.rename(columns={'Adj Close': 'AdjClose'}, inplace=True)
        
        # Convert to datetime index if needed
        if not isinstance(result.index, pd.DatetimeIndex):
            if 'Date' in result.columns:
                result = result.set_index('Date')
            result.index = pd.to_datetime(result.index)
        
        # Sort by date
        result = result.sort_index()
        
        # Fix dividend adjusted prices (check for sudden price drops)
        if 'Close' in result.columns and 'AdjClose' in result.columns:
            # Use AdjClose instead of Close
            result['Close'] = result['AdjClose']
        
        # Check for duplicate dates
        if result.index.duplicated().any():
            logger.warning(f"Removing {result.index.duplicated().sum()} duplicate dates")
            result = result[~result.index.duplicated(keep='first')]
        
        # Check for outliers in price (extreme values)
        for col in ['Open', 'High', 'Low', 'Close', 'AdjClose']:
            if col in result.columns:
                # Get median and IQR
                median = result[col].median()
                q1 = result[col].quantile(0.25)
                q3 = result[col].quantile(0.75)
                iqr = q3 - q1
                
                # Fix extreme outliers (20+ IQR from median)
                lower_bound = q1 - (20 * iqr)
                upper_bound = q3 + (20 * iqr)
                extreme_mask = (result[col] < lower_bound) | (result[col] > upper_bound)
                
                if extreme_mask.any():
                    logger.warning(f"Fixing {extreme_mask.sum()} extreme values in {col}")
                    # Replace with previous value
                    result.loc[extreme_mask, col] = np.nan
                    result[col] = result[col].fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Fixed Yahoo Finance data: {len(result)} rows")
        return result
    
    except Exception as e:
        logger.error(f"Error fixing Yahoo Finance data: {str(e)}")
        return df 