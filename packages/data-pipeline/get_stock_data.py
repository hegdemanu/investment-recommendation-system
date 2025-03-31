#!/usr/bin/env python3
"""
Stock Data Downloader
Downloads historical stock data from Yahoo Finance
"""

import os
import sys
import pandas as pd
import yfinance as yf
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import time

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_stock_data(ticker, years=5, interval='1d'):
    """
    Download stock data for a specific ticker
    
    Args:
        ticker (str): Stock ticker symbol
        years (int): Number of years of data to download
        interval (str): Data interval ('1d', '1wk', '1mo')
        
    Returns:
        pandas.DataFrame: Stock data
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        
        logger.info(f"Downloading {ticker} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Download data
        df = stock.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            logger.warning(f"No data found for {ticker}")
            return None
        
        logger.info(f"Downloaded {len(df)} rows of data for {ticker}")
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Get company info
        try:
            info = stock.info
            company_name = info.get('shortName', ticker)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            logger.info(f"Retrieved company info for {ticker}: {company_name} ({sector}/{industry})")
        except Exception as e:
            logger.warning(f"Could not retrieve company info for {ticker}: {e}")
            company_name = ticker
            sector = "Unknown"
            industry = "Unknown"
        
        # Add metadata
        df['Ticker'] = ticker
        df['CompanyName'] = company_name
        df['Sector'] = sector
        df['Industry'] = industry
        
        return df
    
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}")
        return None

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for stock data
    
    Args:
        df (pandas.DataFrame): Stock data with OHLC columns
        
    Returns:
        pandas.DataFrame: Stock data with technical indicators
    """
    if df is None or df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI (14-period)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20-day, 2 standard deviations)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # Volume moving average
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
    
    # Percentage change
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    # 20-day volatility (standard deviation of returns)
    df['Volatility_20d'] = df['Daily_Return'].rolling(window=20).std()
    
    return df

def save_stock_data(df, ticker, output_dir="data"):
    """
    Save stock data to CSV and pickle files
    
    Args:
        df (pandas.DataFrame): Stock data
        ticker (str): Stock ticker symbol
        output_dir (str): Directory to save data
    """
    if df is None or df.empty:
        logger.warning(f"No data to save for {ticker}")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create ticker directory
    ticker_path = output_path / ticker
    ticker_path.mkdir(exist_ok=True)
    
    # Save as CSV
    csv_path = ticker_path / f"{ticker}_data.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV data to {csv_path}")
    
    # Save as pickle for faster loading
    pickle_path = ticker_path / f"{ticker}_data.pkl"
    df.to_pickle(pickle_path)
    logger.info(f"Saved pickle data to {pickle_path}")
    
    # Save metadata
    metadata = {
        'ticker': ticker,
        'company_name': df['CompanyName'].iloc[0] if 'CompanyName' in df.columns else ticker,
        'sector': df['Sector'].iloc[0] if 'Sector' in df.columns else "Unknown",
        'industry': df['Industry'].iloc[0] if 'Industry' in df.columns else "Unknown",
        'data_start': df['Date'].min().strftime('%Y-%m-%d') if 'Date' in df.columns else "Unknown",
        'data_end': df['Date'].max().strftime('%Y-%m-%d') if 'Date' in df.columns else "Unknown",
        'rows': len(df),
        'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save metadata as JSON
    import json
    with open(ticker_path / f"{ticker}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
def main():
    parser = argparse.ArgumentParser(description='Stock Data Downloader')
    parser.add_argument('--tickers', type=str, required=True, 
                        help='Stock ticker symbols (comma-separated for multiple)')
    parser.add_argument('--years', type=int, default=5, 
                        help='Number of years of historical data to download')
    parser.add_argument('--interval', type=str, default='1d', choices=['1d', '1wk', '1mo'],
                        help='Data interval (1d=daily, 1wk=weekly, 1mo=monthly)')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory for saved data')
    parser.add_argument('--indicators', action='store_true',
                        help='Calculate technical indicators')
    args = parser.parse_args()
    
    # Parse tickers
    tickers = [ticker.strip() for ticker in args.tickers.split(',')]
    
    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Download data for each ticker
    for ticker in tickers:
        # Check for Indian stock tickers that need .NS or .BO suffix
        if not (ticker.endswith('.NS') or ticker.endswith('.BO') or ticker.endswith('.NSE') or ticker.endswith('.BSE')):
            logger.info(f"Processing ticker: {ticker}")
        else:
            logger.info(f"Processing Indian stock: {ticker}")
        
        # Download data
        df = download_stock_data(ticker, years=args.years, interval=args.interval)
        
        if df is not None and not df.empty:
            # Calculate technical indicators if requested
            if args.indicators:
                logger.info(f"Calculating technical indicators for {ticker}")
                df = calculate_technical_indicators(df)
            
            # Save data
            save_stock_data(df, ticker, output_dir=args.output)
        
        # Delay to avoid hitting API rate limits
        if len(tickers) > 1:
            time.sleep(1)
    
    logger.info(f"Completed downloading data for {len(tickers)} tickers")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1) 