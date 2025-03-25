import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
from typing import List, Dict, Optional
import os

class DataAcquisition:
    """
    Module for acquiring financial data from various sources.
    Supports multiple data providers and formats.
    """
    
    def __init__(self, data_dir: str = "./data"):
        """
        Initialize the DataAcquisition module.
        
        Parameters:
        -----------
        data_dir : str, optional
            Directory to store downloaded data
        """
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created data directory: {data_dir}")
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fetch_stock_data(self, 
                        tickers: List[str], 
                        start_date: str, 
                        end_date: Optional[str] = None,
                        source: str = "yfinance") -> pd.DataFrame:
        """
        Fetch historical stock data for multiple tickers.
        
        Parameters:
        -----------
        tickers : List[str]
            List of stock tickers
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format (defaults to today)
        source : str, optional
            Data source to use ("yfinance" or "alpha_vantage")
            
        Returns:
        --------
        pd.DataFrame : Historical stock data
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        if source == "yfinance":
            return self._fetch_from_yfinance(tickers, start_date, end_date)
        elif source == "alpha_vantage":
            return self._fetch_from_alpha_vantage(tickers, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    def _fetch_from_yfinance(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        all_data = []
        
        for ticker in tickers:
            try:
                # Add .NS suffix for Indian stocks
                if not ticker.endswith('.NS'):
                    ticker = f"{ticker}.NS"
                
                print(f"Fetching data for {ticker}...")
                stock = yf.Ticker(ticker)
                
                # Get historical data
                df = stock.history(start=start_date, end=end_date)
                
                if not df.empty:
                    # Reset index to make Date a column
                    df = df.reset_index()
                    
                    # Add ticker column
                    df['ticker'] = ticker.replace('.NS', '')
                    
                    # Rename columns to match our expected format
                    df.rename(columns={
                        'Date': 'Date',
                        'Open': 'Open',
                        'High': 'High',
                        'Low': 'Low',
                        'Close': 'Price',
                        'Volume': 'Volume'
                    }, inplace=True)
                    
                    all_data.append(df)
                    print(f"Successfully fetched {len(df)} records for {ticker}")
                else:
                    print(f"No data found for {ticker}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()
    
    def _fetch_from_alpha_vantage(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage API."""
        # Implementation for Alpha Vantage API
        # You'll need to add your Alpha Vantage API key
        pass
    
    def fetch_fundamental_data(self, ticker: str) -> Dict:
        """
        Fetch fundamental data for a stock.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
            
        Returns:
        --------
        dict : Fundamental data
        """
        try:
            if not ticker.endswith('.NS'):
                ticker = f"{ticker}.NS"
            
            stock = yf.Ticker(ticker)
            
            # Get company info
            info = stock.info
            
            # Extract relevant fundamental metrics
            fundamental_data = {
                'ticker': ticker.replace('.NS', ''),
                'pe_ratio': info.get('forwardPE', None),
                'peg_ratio': info.get('pegRatio', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'cash_flow': info.get('freeCashflow', None),
                'market_cap': info.get('marketCap', None),
                'dividend_yield': info.get('dividendYield', None),
                'beta': info.get('beta', None),
                'sector': info.get('sector', None),
                'industry': info.get('industry', None)
            }
            
            return fundamental_data
            
        except Exception as e:
            self.logger.error(f"Error fetching fundamental data for {ticker}: {str(e)}")
            return {}
    
    def fetch_market_indices(self, indices: List[str] = ['^NSEI', '^BSESN']) -> pd.DataFrame:
        """
        Fetch data for market indices.
        
        Parameters:
        -----------
        indices : List[str], optional
            List of index symbols
            
        Returns:
        --------
        pd.DataFrame : Index data
        """
        all_data = []
        
        for index in indices:
            try:
                print(f"Fetching data for {index}...")
                index_data = yf.Ticker(index)
                df = index_data.history(period="1y")
                
                if not df.empty:
                    df = df.reset_index()
                    df['index'] = index
                    all_data.append(df)
                    print(f"Successfully fetched {len(df)} records for {index}")
                else:
                    print(f"No data found for {index}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching data for {index}: {str(e)}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, filename: str):
        """
        Save data to CSV file.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to save
        filename : str
            Name of the file to save
        """
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        filename : str
            Name of the file to load
            
        Returns:
        --------
        pd.DataFrame : Loaded data
        """
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            raise FileNotFoundError(f"Data file not found: {filepath}") 