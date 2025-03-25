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
import glob
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataProcessor')

class DataProcessor:
    """
    Module for data loading, cleaning, and preprocessing.
    Supports both CSV/Excel file uploads and API data fetching.
    """
    
    def __init__(self):
        """Initialize the DataProcessor module."""
        self.data = None
        self.market_sentiment = None
        # Define required columns for different data types
        self.stock_required_columns = ['Date', 'Price']
        self.stock_optional_columns = ['Open', 'High', 'Low', 'Vol.', 'Change %']
        self.mf_required_columns = ['Date', 'NAV']
    
    def validate_directory_structure(self, base_dir='./data'):
        """
        Ensure all necessary directories exist, create them if they don't.
        
        Parameters:
        -----------
        base_dir : str
            Base directory for data storage
            
        Returns:
        --------
        bool : True if successful
        """
        directories = [
            f'{base_dir}/raw',
            f'{base_dir}/processed',
            f'{base_dir}/uploads',
            './models',
            './results'
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                    logger.info(f"Created directory: {directory}")
                except Exception as e:
                    logger.error(f"Failed to create directory {directory}: {str(e)}")
                    return False
        
        return True

    def load_and_combine_files(self, pattern, ticker_column=None, default_ticker=None):
        """
        Load multiple CSV/Excel files matching a pattern and combine them.
        Handles missing files gracefully.
        
        Parameters:
        -----------
        pattern : str
            Glob pattern to match files, e.g., './data/raw/*.csv'
        ticker_column : str, optional
            Name of the column containing ticker symbols. If None, will try to infer from data
        default_ticker : str, optional
            Default ticker to use if one cannot be determined from the file
            
        Returns:
        --------
        pd.DataFrame : Combined data from all files
        """
        logger.info(f"Looking for files matching pattern: {pattern}")
        file_paths = glob.glob(pattern)
        
        if not file_paths:
            logger.warning(f"No files found matching pattern: {pattern}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(file_paths)} files matching pattern")
        
        all_data = []
        for file_path in file_paths:
            try:
                # Load data
                df = self.load_file(file_path, ticker_column, default_ticker)
                
                if df is not None and not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
        
        if not all_data:
            logger.warning("No valid data found in any files")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data from {len(all_data)} files, total rows: {len(combined_data)}")
        
        return combined_data
    
    def load_file(self, file_path, ticker_column=None, default_ticker=None):
        """
        Load investment data from a CSV or Excel file with enhanced error handling.
        
        Parameters:
        -----------
        file_path : str
            Path to the file containing stock/mutual fund data
        ticker_column : str, optional
            Name of the column containing ticker symbols
        default_ticker : str, optional
            Default ticker to use if one cannot be determined from the file
            
        Returns:
        --------
        pd.DataFrame : Cleaned and validated data
        """
        logger.info(f"Loading data from {file_path}...")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None
        
        try:
            # Determine file type from extension
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension in ['.xlsx', '.xls']:
                # Excel file
                data = pd.read_excel(file_path)
                logger.info(f"Loaded Excel file with {len(data)} rows and {len(data.columns)} columns.")
            else:
                # CSV file (default) - Try different encodings if default fails
                encodings = ['utf-8', 'latin1', 'ISO-8859-1']
                
                for encoding in encodings:
                    try:
                        # Determine the separator (comma, semicolon, tab)
                        with open(file_path, 'r', encoding=encoding) as f:
                            first_line = f.readline()
                        
                        if ',' in first_line:
                            separator = ','
                        elif ';' in first_line:
                            separator = ';'
                        elif '\t' in first_line:
                            separator = '\t'
                        else:
                            separator = ','
                        
                        data = pd.read_csv(file_path, sep=separator, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        if encoding == encodings[-1]:
                            raise
                        continue
                
                logger.info(f"Successfully loaded CSV file with {len(data)} rows and {len(data.columns)} columns.")
            
            # Handle ticker column
            self._assign_ticker(data, file_path, ticker_column, default_ticker)
            
            # Clean and validate data
            data = self._clean_data(data)
            
            self.data = data
            return data
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            return None
    
    def _assign_ticker(self, data, file_path, ticker_column=None, default_ticker=None):
        """
        Assign ticker values to the dataframe based on available information.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The dataframe to modify
        file_path : str
            Path to the source file
        ticker_column : str, optional
            Name of column containing ticker data
        default_ticker : str, optional
            Default ticker to use if none can be determined
        """
        # If ticker column already exists and has values, nothing to do
        if 'ticker' in data.columns and not data['ticker'].isna().all():
            return
        
        # If specified ticker column exists, use it
        if ticker_column and ticker_column in data.columns:
            logger.info(f"Using {ticker_column} as ticker column")
            data['ticker'] = data[ticker_column]
            return
        
        # Try to find a column that might contain ticker information
        possible_ticker_columns = ['ticker', 'symbol', 'stock', 'security', 'instrument']
        for col in possible_ticker_columns:
            if col in data.columns and not data[col].isna().all():
                logger.info(f"Using column '{col}' as ticker source")
                data['ticker'] = data[col]
                return
        
        # If we couldn't find a ticker column, use the default or derive from filename
        if default_ticker:
            ticker = default_ticker
        else:
            # Try to extract from filename as last resort
            file_name = os.path.basename(file_path)
            ticker = os.path.splitext(file_name)[0]
        
        logger.info(f"Using '{ticker}' as default ticker for all rows")
        data['ticker'] = ticker
    
    def _clean_data(self, data):
        """
        Clean and validate data, handling missing values and column format issues.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw data to clean
            
        Returns:
        --------
        pd.DataFrame : Cleaned data
        """
        if data is None or data.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        data = data.copy()
        
        # Standardize column names (strip whitespace, check case variations)
        data.columns = [col.strip() for col in data.columns]
        
        # Handle common column name variations
        column_mappings = {
            'date': 'Date',
            'datetime': 'Date',
            'time': 'Date',
            'timestamp': 'Date',
            'close': 'Price',
            'closing price': 'Price',
            'price': 'Price',
            'last price': 'Price',
            'open': 'Open',
            'opening price': 'Open',
            'high': 'High',
            'highest price': 'High',
            'low': 'Low',
            'lowest price': 'Low',
            'volume': 'Vol.',
            'vol': 'Vol.',
            'turnover': 'Vol.',
            'change': 'Change %',
            'change%': 'Change %',
            'pct change': 'Change %',
            'mutual fund nav': 'NAV',
            'nav': 'NAV',
            'net asset value': 'NAV'
        }
        
        # Standardize column names
        for old_col, new_col in column_mappings.items():
            if old_col in data.columns:
                data = data.rename(columns={old_col: new_col})
        
        # Identify data type (stock or mutual fund)
        is_mf = 'NAV' in data.columns
        
        # Check for required columns
        required_cols = self.mf_required_columns if is_mf else self.stock_required_columns
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            
            # Try to infer missing columns if possible
            if 'Price' in missing_cols and 'NAV' in data.columns:
                logger.info("Using NAV as Price")
                data['Price'] = data['NAV']
                missing_cols.remove('Price')
            
            # If still missing critical columns, return empty DataFrame
            if 'Date' in missing_cols or ('Price' in missing_cols and 'NAV' not in data.columns):
                logger.error("Missing critical columns, cannot process data")
                return pd.DataFrame()
        
        # Handle Date column
        if 'Date' in data.columns:
            try:
                # Try to convert Date to datetime format
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                
                # Drop rows with invalid dates
                invalid_dates = data['Date'].isna()
                if invalid_dates.any():
                    logger.warning(f"Dropped {invalid_dates.sum()} rows with invalid dates")
                    data = data[~invalid_dates]
                
                # Sort by date
                data = data.sort_values('Date')
            except Exception as e:
                logger.error(f"Error processing Date column: {str(e)}")
        
        # Special handling for volume column with K, M, B suffixes
        if 'Vol.' in data.columns:
            try:
                # Function to convert volume strings like "1.5M" to numeric values
                def convert_volume(vol):
                    if pd.isna(vol):
                        return np.nan
                    if isinstance(vol, (int, float)):
                        return float(vol)
                    
                    vol = str(vol).replace(',', '')
                    
                    if 'K' in vol:
                        return float(vol.replace('K', '')) * 1000
                    elif 'M' in vol:
                        return float(vol.replace('M', '')) * 1000000
                    elif 'B' in vol:
                        return float(vol.replace('B', '')) * 1000000000
                    else:
                        return float(vol)
                
                data['Vol.'] = data['Vol.'].apply(convert_volume)
                logger.info("Processed volume column with K/M/B suffixes")
            except Exception as e:
                logger.error(f"Error processing volume column: {str(e)}")
                # If conversion fails, set to NaN
                data['Vol.'] = np.nan
        
        # Handle numeric columns
        numeric_cols = ['Price', 'Open', 'High', 'Low', 'NAV']
        for col in numeric_cols:
            if col in data.columns:
                # Replace commas in numeric values
                if data[col].dtype == object:
                    data[col] = data[col].astype(str).str.replace(',', '')
                # Convert to numeric, coerce errors to NaN
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Handle percent columns
        if 'Change %' in data.columns:
            try:
                if data['Change %'].dtype == object:
                    # Remove % symbol and convert to numeric
                    data['Change %'] = data['Change %'].astype(str).str.replace('%', '')
                data['Change %'] = pd.to_numeric(data['Change %'], errors='coerce')
            except Exception as e:
                logger.error(f"Error processing Change % column: {str(e)}")
                # If conversion fails, calculate from Price
                if 'Price' in data.columns:
                    logger.info("Calculating Change % from Price")
                    data['Change %'] = data['Price'].pct_change() * 100
        
        # Calculate missing fields if possible
        if 'Price' in data.columns and 'Change %' not in data.columns:
            logger.info("Calculating Change % from Price")
            data['Change %'] = data['Price'].pct_change() * 100
        
        # Fill missing values for technical analysis columns
        for col in ['Open', 'High', 'Low']:
            if col not in data.columns and 'Price' in data.columns:
                logger.info(f"Creating {col} column from Price")
                data[col] = data['Price']
        
        # Handle any remaining NaN values
        # For Price/NAV, forward/backward fill
        if 'Price' in data.columns:
            data['Price'] = data['Price'].fillna(method='ffill').fillna(method='bfill')
        
        if 'NAV' in data.columns:
            data['NAV'] = data['NAV'].fillna(method='ffill').fillna(method='bfill')
        
        # For other columns, use more conservative approaches
        for col in ['Open', 'High', 'Low']:
            if col in data.columns:
                # For these, use Price as fallback
                if 'Price' in data.columns:
                    data[col] = data[col].fillna(data['Price'])
        
        if 'Vol.' in data.columns:
            # For volume, fill with median to avoid skewing
            data['Vol.'] = data['Vol.'].fillna(data['Vol.'].median())
        
        # Drop any remaining rows with critical NaN values
        critical_cols = ['Date', 'Price'] if not is_mf else ['Date', 'NAV']
        data = data.dropna(subset=critical_cols)
        
        logger.info(f"After cleaning, data has {len(data)} rows")
        return data
    
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
        logger.info(f"Fetching data for {len(ticker_list)} tickers from API...")
        
        all_data = []
        
        for ticker in ticker_list:
            logger.info(f"Processing ticker: {ticker}")
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
                            'volume': 'Vol.',
                            'date': 'Date'
                        }, inplace=True)
                        
                        # Add ticker column
                        df['ticker'] = ticker
                        
                        # Calculate daily change percentage
                        df['Change %'] = df['Price'].pct_change() * 100
                        
                        # Clean the data
                        df = self._clean_data(df)
                        
                        if not df.empty:
                            all_data.append(df)
                            logger.info(f"Successfully retrieved {len(df)} records for {ticker}")
                        else:
                            logger.warning(f"Data processing resulted in empty dataset for {ticker}")
                    else:
                        logger.warning(f"No price data found for {ticker}")
                else:
                    logger.error(f"Failed to fetch data for {ticker}: Status code {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {str(e)}")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            self.data = combined_data
            logger.info(f"Successfully loaded data with {len(combined_data)} rows for {len(ticker_list)} tickers.")
            return combined_data
        else:
            logger.warning("Failed to fetch any data from API.")
            return pd.DataFrame()
    
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
        Preprocess data by calculating technical indicators, handling missing values,
        and ensuring data is ready for model training.
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            Data to preprocess. If None, uses self.data
            
        Returns:
        --------
        pd.DataFrame : Preprocessed data
        """
        if data is None:
            data = self.data
        
        if data is None or data.empty:
            logger.warning("No data available for preprocessing")
            return pd.DataFrame()
        
        try:
            # Make a copy to avoid modifying the original
            processed_data = data.copy()
            
            # Ensure data is sorted by date
            if 'Date' in processed_data.columns:
                processed_data = processed_data.sort_values(['ticker', 'Date'])
            
            # Check if data has required columns for technical indicator calculation
            required_columns = ['Date', 'Price']
            if not all(col in processed_data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in processed_data.columns]
                logger.warning(f"Missing required columns for preprocessing: {missing}")
                
                # Try to recover - if NAV is present but Price is not
                if 'Price' in missing and 'NAV' in processed_data.columns:
                    logger.info("Using NAV as Price for preprocessing")
                    processed_data['Price'] = processed_data['NAV']
                    missing.remove('Price')
                
                # If still missing critical columns, return what we have
                if missing:
                    logger.error("Cannot proceed with preprocessing due to missing critical columns")
                    return processed_data
            
            # Calculate technical indicators if possible
            try:
                processed_data = self.calculate_technical_indicators(processed_data)
            except Exception as e:
                logger.error(f"Error calculating technical indicators: {str(e)}")
            
            # Merge sentiment data if available
            if self.market_sentiment is not None:
                try:
                    processed_data = self.merge_sentiment_with_data(processed_data, self.market_sentiment)
                except Exception as e:
                    logger.error(f"Error merging sentiment data: {str(e)}")
            
            # Handle any remaining missing values
            numeric_cols = processed_data.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_cols:
                if processed_data[col].isna().any():
                    # For most indicators, use forward fill then backward fill
                    fill_method = 'ffill'
                    processed_data[col] = processed_data[col].fillna(method=fill_method)
                    
                    # If still has NaN, use backward fill
                    if processed_data[col].isna().any():
                        processed_data[col] = processed_data[col].fillna(method='bfill')
                    
                    # If still has NaN, use column median
                    if processed_data[col].isna().any():
                        processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                    
                    logger.info(f"Filled missing values in column: {col}")
            
            # Drop any rows that still have NaN in critical columns
            critical_cols = ['Date', 'Price']
            if processed_data[critical_cols].isna().any().any():
                before_count = len(processed_data)
                processed_data = processed_data.dropna(subset=critical_cols)
                after_count = len(processed_data)
                logger.info(f"Dropped {before_count - after_count} rows with missing values in critical columns")
            
            logger.info(f"Preprocessing complete. Final data shape: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            return data  # Return original data on error 