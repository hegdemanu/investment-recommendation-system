"""
Data Processing Module for Investment Recommendation System
Handles data loading, cleaning, and preprocessing with enhanced metrics
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
import json


class DataProcessor:
    """
    Module for data loading, cleaning, and preprocessing.
    Supports both CSV file uploads and API data fetching.
    """
    
    def __init__(self):
        """Initialize the DataProcessor module."""
        self.data = None
        self.market_sentiment = None
    
    def load_from_csv(self, file_path):
        """
        Load investment data from a CSV file.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing stock/mutual fund data
            
        Returns:
        --------
        pd.DataFrame : Loaded data
        """
        print(f"Loading data from {file_path}...")
        try:
            data = pd.read_csv(file_path)
            self.data = data
            print(f"Successfully loaded data with {len(data)} rows and {len(data.columns)} columns.")
            return data
        except Exception as e:
            print(f"Error loading CSV file: {str(e)}")
            return None
    
    def load_from_api(self, ticker_list, start_date, end_date, api_key):
        """
        Load investment data from RapidAPI.
        
        Parameters:
        -----------
        ticker_list : list
            List of stock/MF tickers to fetch data for
        start_date : str
            Start date for data retrieval (YYYY-MM-DD)
        end_date : str
            End date for data retrieval (YYYY-MM-DD)
        api_key : str
            API key for RapidAPI
            
        Returns:
        --------
        pd.DataFrame : Loaded data
        """
        print(f"Fetching data for {len(ticker_list)} tickers from API...")
        
        all_data = []
        
        for ticker in ticker_list:
            print(f"Processing ticker: {ticker}")
            url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v3/get-historical-data"
            
            querystring = {"symbol": f"{ticker}.NS", "region": "IN"}  # .NS for NSE India
            
            headers = {
                'x-rapidapi-key': api_key,
                'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
            }
            
            try:
                response = requests.request("GET", url, headers=headers, params=querystring)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'prices' in data and len(data['prices']) > 0:
                        df = pd.DataFrame(data['prices'])
                        
                        # Convert timestamp to datetime
                        df['date'] = pd.to_datetime(df['date'], unit='s')
                        
                        # Filter by date range
                        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                        
                        # Rename columns to match our expected format
                        df.rename(columns={
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Price',
                            'volume': 'Vol.'
                        }, inplace=True)
                        
                        # Add ticker column
                        df['ticker'] = ticker
                        
                        # Calculate daily change percentage
                        df['Change %'] = df['Price'].pct_change() * 100
                        
                        all_data.append(df)
                        print(f"Successfully retrieved {len(df)} records for {ticker}")
                    else:
                        print(f"No price data found for {ticker}")
                else:
                    print(f"Failed to fetch data for {ticker}: Status code {response.status_code}")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            self.data = combined_data
            print(f"Successfully loaded data with {len(combined_data)} rows for {len(ticker_list)} tickers.")
            return combined_data
        else:
            print("Failed to fetch any data from API.")
            return None
    
    def load_fundamental_data(self, ticker_list, api_key):
        """
        Load fundamental financial data for tickers.
        
        Parameters:
        -----------
        ticker_list : list
            List of stock tickers to fetch fundamentals for
        api_key : str
            API key for RapidAPI
            
        Returns:
        --------
        pd.DataFrame : Fundamental data
        """
        print(f"Fetching fundamental data for {len(ticker_list)} tickers...")
        
        fundamental_data = []
        
        for ticker in ticker_list:
            print(f"Fetching fundamentals for {ticker}...")
            url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/stock/v2/get-financials"
            
            querystring = {"symbol": f"{ticker}.NS", "region": "IN"}
            
            headers = {
                'x-rapidapi-key': api_key,
                'x-rapidapi-host': "apidojo-yahoo-finance-v1.p.rapidapi.com"
            }
            
            try:
                response = requests.request("GET", url, headers=headers, params=querystring)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract key financial metrics
                    metrics = {
                        'ticker': ticker,
                        'date': datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    # Balance sheet metrics
                    if 'balanceSheetHistory' in data and 'balanceSheetStatements' in data['balanceSheetHistory']:
                        statements = data['balanceSheetHistory']['balanceSheetStatements']
                        if statements:
                            latest = statements[0]
                            
                            # Extract total debt
                            if 'totalDebt' in latest and 'raw' in latest['totalDebt']:
                                metrics['total_debt'] = latest['totalDebt']['raw']
                            
                            # Extract total equity
                            if 'totalStockholderEquity' in latest and 'raw' in latest['totalStockholderEquity']:
                                metrics['total_equity'] = latest['totalStockholderEquity']['raw']
                                
                                # Calculate debt-to-equity ratio
                                if 'total_debt' in metrics:
                                    metrics['debt_equity_ratio'] = metrics['total_debt'] / metrics['total_equity']
                    
                    # Cash flow metrics
                    if 'cashflowStatementHistory' in data and 'cashflowStatements' in data['cashflowStatementHistory']:
                        statements = data['cashflowStatementHistory']['cashflowStatements']
                        if len(statements) >= 2:  # Need at least 2 statements to calculate growth
                            latest = statements[0]
                            previous = statements[1]
                            
                            # Extract operating cash flow
                            if 'operatingCashflow' in latest and 'raw' in latest['operatingCashflow']:
                                metrics['operating_cash_flow'] = latest['operatingCashflow']['raw']
                            
                            # Calculate cash flow growth
                            if ('operatingCashflow' in latest and 'raw' in latest['operatingCashflow'] and
                                'operatingCashflow' in previous and 'raw' in previous['operatingCashflow']):
                                latest_cf = latest['operatingCashflow']['raw']
                                previous_cf = previous['operatingCashflow']['raw']
                                
                                if previous_cf != 0:
                                    metrics['cash_flow_growth'] = ((latest_cf / previous_cf) - 1) * 100
                    
                    # Income statement metrics
                    if 'earnings' in data and 'financialsChart' in data['earnings']:
                        if 'yearly' in data['earnings']['financialsChart']:
                            yearly_data = data['earnings']['financialsChart']['yearly']
                            if yearly_data:
                                latest = yearly_data[-1]
                                if 'earnings' in latest and 'raw' in latest['earnings']:
                                    metrics['earnings'] = latest['earnings']['raw']
                                if 'revenue' in latest and 'raw' in latest['revenue']:
                                    metrics['revenue'] = latest['revenue']['raw']
                    
                    # EPS metrics
                    if 'defaultKeyStatistics' in data:
                        stats = data['defaultKeyStatistics']
                        
                        # Trailing EPS
                        if 'trailingEps' in stats and 'raw' in stats['trailingEps']:
                            metrics['eps'] = stats['trailingEps']['raw']
                        
                        # Forward EPS
                        if 'forwardEps' in stats and 'raw' in stats['forwardEps']:
                            metrics['forward_eps'] = stats['forwardEps']['raw']
                        
                        # PEG Ratio
                        if 'pegRatio' in stats and 'raw' in stats['pegRatio']:
                            metrics['peg_ratio'] = stats['pegRatio']['raw']
                    
                    # Current price for PE ratio calculation
                    if 'price' in data and 'regularMarketPrice' in data['price'] and 'raw' in data['price']['regularMarketPrice']:
                        metrics['current_price'] = data['price']['regularMarketPrice']['raw']
                        
                        # Calculate PE Ratio if EPS is available
                        if 'eps' in metrics and metrics['eps'] != 0:
                            metrics['pe_ratio'] = metrics['current_price'] / metrics['eps']
                    
                    fundamental_data.append(metrics)
                    print(f"Successfully retrieved fundamental data for {ticker}")
                else:
                    print(f"Failed to fetch fundamental data for {ticker}: Status code {response.status_code}")
            except Exception as e:
                print(f"Error fetching fundamental data for {ticker}: {str(e)}")
        
        if fundamental_data:
            return pd.DataFrame(fundamental_data)
        else:
            print("Failed to fetch any fundamental data.")
            return pd.DataFrame()
    
    def fetch_market_sentiment(self, api_key=None):
        """
        Fetch market sentiment data (fear/greed index).
        
        Parameters:
        -----------
        api_key : str, optional
            API key if required
            
        Returns:
        --------
        dict : Market sentiment data
        """
        print("Fetching market sentiment (fear/greed) data...")
        
        try:
            # Attempt to fetch from CNN Fear & Greed API (or alternative)
            # For demonstration, we're using a simulated response
            
            # In a real implementation, you would use:
            # url = "https://fear-and-greed-index.p.rapidapi.com/v1/fgi"
            # headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "fear-and-greed-index.p.rapidapi.com"}
            # response = requests.get(url, headers=headers)
            # sentiment_data = response.json()
            
            # Simulated response
            sentiment_data = {
                'fear_greed_index': 65,  # Range 0-100
                'fear_greed_classification': 'Greed',  # Extreme Fear, Fear, Neutral, Greed, Extreme Greed
                'previous_close': 62,
                'one_week_ago': 55,
                'one_month_ago': 45,
                'one_year_ago': 70,
                'timestamp': datetime.now().strftime("%Y-%m-%d")
            }
            
            self.market_sentiment = sentiment_data
            print(f"Current Fear & Greed Index: {sentiment_data['fear_greed_index']} ({sentiment_data['fear_greed_classification']})")
            return sentiment_data
        
        except Exception as e:
            print(f"Error fetching market sentiment data: {str(e)}")
            
            # Fallback to a neutral sentiment
            fallback_sentiment = {
                'fear_greed_index': 50,
                'fear_greed_classification': 'Neutral',
                'previous_close': 50,
                'one_week_ago': 50,
                'one_month_ago': 50,
                'one_year_ago': 50,
                'timestamp': datetime.now().strftime("%Y-%m-%d"),
                'is_fallback': True
            }
            
            self.market_sentiment = fallback_sentiment
            print("Using fallback neutral sentiment data.")
            return fallback_sentiment
    
    def save_sentiment_history(self, sentiment_data, file_path='./data/sentiment_history.json'):
        """
        Save sentiment data to history file.
        
        Parameters:
        -----------
        sentiment_data : dict
            Sentiment data to save
        file_path : str, optional
            Path to save sentiment history
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Load existing history if available
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    history = json.load(f)
            except:
                history = []
        else:
            history = []
        
        # Add current sentiment data with timestamp
        sentiment_with_time = sentiment_data.copy()
        sentiment_with_time['recorded_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to history and save
        history.append(sentiment_with_time)
        
        with open(file_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Saved sentiment data to history file: {file_path}")
    
    def calculate_technical_indicators(self, data):
        """
        Calculate technical indicators for the data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data to calculate indicators for
            
        Returns:
        --------
        pd.DataFrame : Data with technical indicators
        """
        print("Calculating technical indicators...")
        
        df = data.copy()
        
        # Group by ticker if multiple tickers in data
        if 'ticker' in df.columns:
            grouped = df.groupby('ticker')
            result_dfs = []
            
            for ticker, group in grouped:
                group = group.sort_values('Date')
                # Calculate RSI (14-day)
                delta = group['Price'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                group['rsi'] = 100 - (100 / (1 + rs))
                
                # Calculate MACD
                group['ema_12'] = group['Price'].ewm(span=12, adjust=False).mean()
                group['ema_26'] = group['Price'].ewm(span=26, adjust=False).mean()
                group['macd'] = group['ema_12'] - group['ema_26']
                group['macd_signal'] = group['macd'].ewm(span=9, adjust=False).mean()
                group['macd_histogram'] = group['macd'] - group['macd_signal']
                
                # Calculate Bollinger Bands (20-day, 2 standard deviations)
                group['bb_middle'] = group['Price'].rolling(window=20).mean()
                std_dev = group['Price'].rolling(window=20).std()
                group['bb_upper'] = group['bb_middle'] + (std_dev * 2)
                group['bb_lower'] = group['bb_middle'] - (std_dev * 2)
                
                result_dfs.append(group)
            
            df = pd.concat(result_dfs, ignore_index=True)
        else:
            # Calculate for a single series
            df = df.sort_values('Date')
            
            # Calculate RSI (14-day)
            delta = df['Price'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            df['ema_12'] = df['Price'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['Price'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Calculate Bollinger Bands (20-day, 2 standard deviations)
            df['bb_middle'] = df['Price'].rolling(window=20).mean()
            std_dev = df['Price'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (std_dev * 2)
            df['bb_lower'] = df['bb_middle'] - (std_dev * 2)
        
        print("Technical indicators calculation complete.")
        return df
    
    def merge_sentiment_with_data(self, data, sentiment_data):
        """
        Merge sentiment data with price data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
        sentiment_data : dict
            Sentiment data
            
        Returns:
        --------
        pd.DataFrame : Data with sentiment
        """
        df = data.copy()
        
        # Add fear/greed index to all rows
        df['fear_greed_index'] = sentiment_data['fear_greed_index']
        df['fear_greed_classification'] = sentiment_data['fear_greed_classification']
        
        print(f"Added sentiment data to {len(df)} rows: Fear/Greed Index = {sentiment_data['fear_greed_index']}")
        return df
    
    def merge_fundamental_data(self, price_data, fundamental_data):
        """
        Merge fundamental data with price data.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Price data
        fundamental_data : pd.DataFrame
            Fundamental data
            
        Returns:
        --------
        pd.DataFrame : Merged data
        """
        if fundamental_data.empty:
            print("No fundamental data to merge.")
            return price_data
        
        # Ensure both dataframes have 'ticker' column
        if 'ticker' not in price_data.columns or 'ticker' not in fundamental_data.columns:
            print("Both dataframes must have 'ticker' column for merging.")
            return price_data
        
        # Convert columns to appropriate data types
        for col in ['debt_equity_ratio', 'cash_flow_growth', 'pe_ratio', 'peg_ratio']:
            if col in fundamental_data.columns:
                fundamental_data[col] = pd.to_numeric(fundamental_data[col], errors='coerce')
        
        # Get latest data point for each ticker in price_data
        latest_prices = price_data.sort_values('Date').groupby('ticker').last().reset_index()
        
        # Merge fundamental data with latest price data
        merged = latest_prices.merge(fundamental_data, on='ticker', how='left')
        
        print(f"Merged fundamental data for {len(merged)} tickers.")
        return merged
    
    def preprocess(self, data=None):
        """
        Clean and preprocess the data.
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            Data to preprocess. If None, uses self.data
            
        Returns:
        --------
        pd.DataFrame : Preprocessed data
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Please load data first.")
            data = self.data
        
        print("Preprocessing data...")
        
        # Create a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure we have a Date column (standardize column names)
        if 'Date' not in df.columns:
            if 'date' in df.columns:
                df['Date'] = pd.to_datetime(df['date'])
                df.drop('date', axis=1, inplace=True)
            else:
                raise ValueError("No date column found in data.")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                print(f"Filling {df[col].isnull().sum()} missing values in column {col}")
                df[col] = df[col].fillna(df[col].median())
        
        # Calculate additional financial metrics if they don't exist
        if 'Price' in df.columns:
            # Calculate volatility (20-day rolling standard deviation)
            if 'ticker' in df.columns:
                df['Volatility_20d'] = df.groupby('ticker')['Price'].transform(lambda x: x.rolling(window=20).std())
            else:
                df['Volatility_20d'] = df['Price'].rolling(window=20).std()
            
            # Calculate returns (daily and weekly)
            if 'ticker' in df.columns:
                df['Daily_Return'] = df.groupby('ticker')['Price'].transform(lambda x: x.pct_change())
                df['Weekly_Return'] = df.groupby('ticker')['Price'].transform(lambda x: x.pct_change(5))
            else:
                df['Daily_Return'] = df['Price'].pct_change()
                df['Weekly_Return'] = df['Price'].pct_change(5)
        
        # Sort by date and ticker if available
        if 'ticker' in df.columns:
            df = df.sort_values(['ticker', 'Date'])
        else:
            df = df.sort_values('Date')
        
        # Calculate PE Ratio if EPS is available
        if 'EPS' in df.columns and 'Price' in df.columns:
            df['PE Ratio'] = df['Price'] / df['EPS']
        
        # Calculate PEG Ratio if PE Ratio and EPS Growth are available
        if 'PE Ratio' in df.columns and 'EPS Growth' in df.columns:
            df['PEG Ratio'] = df['PE Ratio'] / df['EPS Growth']
        
        # Calculate Sharpe Ratio (annualized return / annualized volatility)
        if 'Daily_Return' in df.columns:
            # Group by ticker if available
            if 'ticker' in df.columns:
                # Calculate mean return and std dev for each ticker
                avg_returns = df.groupby('ticker')['Daily_Return'].mean() * 252  # Annualized
                std_returns = df.groupby('ticker')['Daily_Return'].std() * np.sqrt(252)  # Annualized
                
                # Create Sharpe Ratio for risk-free rate of 5% (typical for India)
                sharpe_ratios = (avg_returns - 0.05) / std_returns
                
                # Map Sharpe Ratio back to each row
                df['Sharpe_Ratio'] = df['ticker'].map(sharpe_ratios)
            else:
                # Calculate for the single asset
                avg_return = df['Daily_Return'].mean() * 252  # Annualized
                std_return = df['Daily_Return'].std() * np.sqrt(252)  # Annualized
                
                # Create Sharpe Ratio for risk-free rate of 5% (typical for India)
                sharpe_ratio = (avg_return - 0.05) / std_return
                
                df['Sharpe_Ratio'] = sharpe_ratio
        
        # Drop rows with NaN values created by the rolling calculations
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows with missing values after calculations")
        
        print(f"Preprocessing complete. Final dataset has {len(df)} rows.")
        return df 