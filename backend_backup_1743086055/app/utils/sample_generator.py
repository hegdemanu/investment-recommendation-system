"""
Sample Data Generator Module

This module provides functions to generate sample data for testing and development
of dashboard components and visualizations.
"""
import json
import os
import random
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)

def generate_stock_price_data(ticker, days=30, start_price=None, volatility=0.02):
    """
    Generate synthetic stock price data for the specified ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days to generate data for
        start_price (float, optional): Starting price, random if None
        volatility (float): Daily price volatility
        
    Returns:
        pandas.DataFrame: DataFrame with OHLC data for the specified ticker
    """
    if start_price is None:
        start_price = random.uniform(100, 1000)
    
    # End date is today, start date is (days) days ago
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    # Generate dates (business days only)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Initialize DataFrame
    df = pd.DataFrame(index=date_range)
    df.index.name = 'Date'
    
    # Generate price data using random walk
    price_series = [start_price]
    for i in range(1, len(date_range)):
        # Random daily return with specified volatility
        daily_return = np.random.normal(0, volatility)
        # New price based on previous price and return
        new_price = price_series[-1] * (1 + daily_return)
        price_series.append(new_price)
    
    # Create OHLC data
    df['Close'] = price_series
    
    # Generate Open, High, Low from Close
    for i in range(len(df)):
        if i == 0:
            df.loc[df.index[i], 'Open'] = df.loc[df.index[i], 'Close'] * (1 - random.uniform(0, 0.01))
        else:
            df.loc[df.index[i], 'Open'] = df.loc[df.index[i-1], 'Close'] * (1 + random.uniform(-0.01, 0.01))
        
        # High is higher than both Open and Close
        high_pct = random.uniform(0.005, 0.02)
        df.loc[df.index[i], 'High'] = max(df.loc[df.index[i], 'Open'], df.loc[df.index[i], 'Close']) * (1 + high_pct)
        
        # Low is lower than both Open and Close
        low_pct = random.uniform(0.005, 0.02)
        df.loc[df.index[i], 'Low'] = min(df.loc[df.index[i], 'Open'], df.loc[df.index[i], 'Close']) * (1 - low_pct)
    
    # Generate Volume
    avg_volume = random.randint(500000, 5000000)
    volume_volatility = 0.3
    df['Volume'] = [int(avg_volume * (1 + np.random.normal(0, volume_volatility))) for _ in range(len(df))]
    
    # Ensure volume is positive
    df['Volume'] = df['Volume'].apply(lambda x: max(x, 10000))
    
    # Reorder columns to OHLCV format
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Add ticker information
    df['Ticker'] = ticker
    
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to stock price data.
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pandas.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    df_with_indicators = df.copy()
    
    # Ensure we have price data
    if 'Close' not in df_with_indicators.columns:
        logger.error("DataFrame does not contain 'Close' column")
        return df
    
    # Simple Moving Averages
    df_with_indicators['SMA_10'] = df_with_indicators['Close'].rolling(window=10).mean()
    df_with_indicators['SMA_20'] = df_with_indicators['Close'].rolling(window=20).mean()
    df_with_indicators['SMA_50'] = df_with_indicators['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df_with_indicators['EMA_10'] = df_with_indicators['Close'].ewm(span=10, adjust=False).mean()
    df_with_indicators['EMA_20'] = df_with_indicators['Close'].ewm(span=20, adjust=False).mean()
    
    # MACD
    df_with_indicators['EMA_12'] = df_with_indicators['Close'].ewm(span=12, adjust=False).mean()
    df_with_indicators['EMA_26'] = df_with_indicators['Close'].ewm(span=26, adjust=False).mean()
    df_with_indicators['MACD'] = df_with_indicators['EMA_12'] - df_with_indicators['EMA_26']
    df_with_indicators['MACD_Signal'] = df_with_indicators['MACD'].ewm(span=9, adjust=False).mean()
    df_with_indicators['MACD_Hist'] = df_with_indicators['MACD'] - df_with_indicators['MACD_Signal']
    
    # RSI (14-period)
    delta = df_with_indicators['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_with_indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20-period, 2 standard deviations)
    df_with_indicators['BB_Middle'] = df_with_indicators['Close'].rolling(window=20).mean()
    df_with_indicators['BB_StdDev'] = df_with_indicators['Close'].rolling(window=20).std()
    df_with_indicators['BB_Upper'] = df_with_indicators['BB_Middle'] + (df_with_indicators['BB_StdDev'] * 2)
    df_with_indicators['BB_Lower'] = df_with_indicators['BB_Middle'] - (df_with_indicators['BB_StdDev'] * 2)
    
    return df_with_indicators

def generate_portfolio_data(num_stocks=10, initial_investment=1000000):
    """
    Generate sample portfolio data.
    
    Args:
        num_stocks (int): Number of stocks in the portfolio
        initial_investment (float): Initial investment amount
        
    Returns:
        dict: Portfolio data with holdings and performance metrics
    """
    # Sample tickers (Indian companies)
    indian_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "ITC.NS",
        "AXISBANK.NS", "LT.NS", "MARUTI.NS", "ASIANPAINT.NS", "SUNPHARMA.NS"
    ]
    
    # Ensure we have enough tickers
    if len(indian_tickers) < num_stocks:
        logger.warning(f"Not enough tickers for {num_stocks} stocks, using available {len(indian_tickers)}")
        num_stocks = len(indian_tickers)
    
    # Select random tickers
    selected_tickers = random.sample(indian_tickers, num_stocks)
    
    # Generate allocation percentages (random but sum to 100%)
    allocation_pcts = np.random.dirichlet(np.ones(num_stocks)) * 100
    
    # Initial investment for each stock
    allocations = allocation_pcts * initial_investment / 100
    
    # Generate current prices and purchase prices
    current_prices = [random.uniform(100, 2000) for _ in range(num_stocks)]
    purchase_prices = [price * random.uniform(0.7, 1.3) for price in current_prices]
    
    # Calculate shares, values, and gains/losses
    shares = [allocation / purchase_price for allocation, purchase_price in zip(allocations, purchase_prices)]
    current_values = [share * price for share, price in zip(shares, current_prices)]
    gains_losses = [value - allocation for value, allocation in zip(current_values, allocations)]
    gains_losses_pct = [100 * (gain / allocation) for gain, allocation in zip(gains_losses, allocations)]
    
    # Create holdings list
    holdings = []
    for i in range(num_stocks):
        holdings.append({
            "ticker": selected_tickers[i],
            "name": f"{selected_tickers[i].split('.')[0]} Limited",
            "allocation_pct": allocation_pcts[i],
            "allocation": allocations[i],
            "shares": shares[i],
            "purchase_price": purchase_prices[i],
            "current_price": current_prices[i],
            "current_value": current_values[i],
            "gain_loss": gains_losses[i],
            "gain_loss_pct": gains_losses_pct[i]
        })
    
    # Calculate total portfolio metrics
    total_current_value = sum(current_values)
    total_gain_loss = sum(gains_losses)
    total_gain_loss_pct = 100 * (total_gain_loss / initial_investment)
    
    # Create portfolio data
    portfolio_data = {
        "initial_investment": initial_investment,
        "current_value": total_current_value,
        "gain_loss": total_gain_loss,
        "gain_loss_pct": total_gain_loss_pct,
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "holdings": holdings
    }
    
    return portfolio_data

def generate_news_data(num_articles=10, days=7):
    """
    Generate sample financial news data.
    
    Args:
        num_articles (int): Number of news articles to generate
        days (int): Number of days to spread news over
        
    Returns:
        list: List of news article dictionaries
    """
    # Sample financial news headlines (mix of positive, negative, and neutral)
    positive_headlines = [
        "Market soars as interest rates stabilize",
        "IT sector booms on strong quarterly results",
        "Government announces major infrastructure package",
        "Banking stocks rally after positive RBI outlook",
        "Auto sector shows signs of recovery as sales surge",
        "Pharmaceutical companies report breakthrough in research",
        "Investment in renewable energy reaches record high",
        "Manufacturing PMI indicates robust growth",
        "FII inflows continue for third straight month",
        "Consumer confidence index reaches 5-year high"
    ]
    
    negative_headlines = [
        "Markets tumble amidst global uncertainty",
        "IT companies face margin pressure due to rising costs",
        "Government debt reaches concerning levels",
        "Banking sector concerns grow as NPAs rise",
        "Auto sales slump continues for second quarter",
        "Pharmaceutical exports hit by regulatory hurdles",
        "Rising inflation dampens economic outlook",
        "Manufacturing sector contracts for third month",
        "FII outflows accelerate as global rates rise",
        "Consumer spending shows signs of slowdown"
    ]
    
    neutral_headlines = [
        "RBI maintains status quo on interest rates",
        "IT companies announce normal quarterly results",
        "Government reviewing fiscal policies for next quarter",
        "Banking regulations under review by central authority",
        "Auto industry awaits policy clarity on electric vehicles",
        "Pharmaceutical sector faces standard regulatory checks",
        "Inflation remains within expected range",
        "Manufacturing sector operates at stable capacity",
        "Foreign investment flows remain balanced",
        "Consumer behavior unchanged in recent survey"
    ]
    
    # Sources for news
    news_sources = [
        "Economic Times", "Business Standard", "Mint", "Financial Express",
        "Bloomberg Quint", "CNBC-TV18", "Business Today", "MoneyControl",
        "Reuters", "Financial Times"
    ]
    
    # Generate random dates within the specified range
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    
    # Create news articles
    news_articles = []
    for _ in range(num_articles):
        # Randomly select headline type and source
        headline_type = random.choice(["positive", "negative", "neutral"])
        if headline_type == "positive":
            headline = random.choice(positive_headlines)
            sentiment = random.uniform(0.6, 1.0)
        elif headline_type == "negative":
            headline = random.choice(negative_headlines)
            sentiment = random.uniform(0.0, 0.4)
        else:
            headline = random.choice(neutral_headlines)
            sentiment = random.uniform(0.4, 0.6)
        
        source = random.choice(news_sources)
        
        # Generate random date within range
        random_days = random.randint(0, days)
        article_date = (end_date - datetime.timedelta(days=random_days)).strftime("%Y-%m-%d")
        
        # Create article object
        news_article = {
            "headline": headline,
            "source": source,
            "date": article_date,
            "url": f"https://example.com/news/{random.randint(10000, 99999)}",
            "sentiment": sentiment,
            "sentiment_label": headline_type.capitalize()
        }
        
        news_articles.append(news_article)
    
    # Sort by date (newest first)
    news_articles.sort(key=lambda x: x["date"], reverse=True)
    
    return news_articles

def generate_dashboard_data():
    """
    Generate complete dashboard sample data.
    
    Returns:
        dict: Complete dashboard dataset
    """
    # Generate market data for major indices
    market_indices = {
        "NIFTY50": generate_stock_price_data("NIFTY50", days=90, start_price=17500, volatility=0.01),
        "SENSEX": generate_stock_price_data("SENSEX", days=90, start_price=58000, volatility=0.01),
        "NIFTYBANK": generate_stock_price_data("NIFTYBANK", days=90, start_price=38000, volatility=0.015),
        "NIFTYIT": generate_stock_price_data("NIFTYIT", days=90, start_price=28000, volatility=0.018)
    }
    
    # Convert DataFrame to dict for JSON serialization
    market_data = {}
    for index, df in market_indices.items():
        market_data[index] = json.loads(df.reset_index().to_json(orient="records", date_format="iso"))
    
    # Generate portfolio holdings data
    portfolio = generate_portfolio_data(num_stocks=8, initial_investment=1000000)
    
    # Generate news data
    news = generate_news_data(num_articles=15, days=10)
    
    # Generate sentiment summary
    sentiment_summary = {
        "overall": random.uniform(0.4, 0.7),
        "by_sector": {
            "Banking": random.uniform(0.3, 0.8),
            "IT": random.uniform(0.3, 0.8),
            "Auto": random.uniform(0.3, 0.8),
            "Pharma": random.uniform(0.3, 0.8),
            "Energy": random.uniform(0.3, 0.8)
        },
        "by_source": {
            "Economic Times": random.uniform(0.3, 0.8),
            "Business Standard": random.uniform(0.3, 0.8),
            "Mint": random.uniform(0.3, 0.8),
            "CNBC-TV18": random.uniform(0.3, 0.8)
        },
        "trend": [
            {"date": (datetime.datetime.now() - datetime.timedelta(days=9)).strftime("%Y-%m-%d"), "value": random.uniform(0.3, 0.8)},
            {"date": (datetime.datetime.now() - datetime.timedelta(days=8)).strftime("%Y-%m-%d"), "value": random.uniform(0.3, 0.8)},
            {"date": (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d"), "value": random.uniform(0.3, 0.8)},
            {"date": (datetime.datetime.now() - datetime.timedelta(days=6)).strftime("%Y-%m-%d"), "value": random.uniform(0.3, 0.8)},
            {"date": (datetime.datetime.now() - datetime.timedelta(days=5)).strftime("%Y-%m-%d"), "value": random.uniform(0.3, 0.8)},
            {"date": (datetime.datetime.now() - datetime.timedelta(days=4)).strftime("%Y-%m-%d"), "value": random.uniform(0.3, 0.8)},
            {"date": (datetime.datetime.now() - datetime.timedelta(days=3)).strftime("%Y-%m-%d"), "value": random.uniform(0.3, 0.8)},
            {"date": (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d"), "value": random.uniform(0.3, 0.8)},
            {"date": (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"), "value": random.uniform(0.3, 0.8)},
            {"date": datetime.datetime.now().strftime("%Y-%m-%d"), "value": random.uniform(0.3, 0.8)}
        ]
    }
    
    # Compile full dashboard data
    dashboard_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "market_data": market_data,
        "portfolio": portfolio,
        "news": news,
        "sentiment": sentiment_summary
    }
    
    return dashboard_data

def create_sample_json_file(filepath, data=None):
    """
    Create a sample JSON file at the specified path.
    
    Args:
        filepath (str): Path to save the JSON file
        data (dict, optional): Custom data to use, or generate random data if None
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Generate data if not provided
        if data is None:
            # Simple sample data structure
            data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "sample": True,
                "metrics": {
                    "accuracy": round(random.uniform(0.75, 0.95), 3),
                    "precision": round(random.uniform(0.75, 0.95), 3),
                    "recall": round(random.uniform(0.75, 0.95), 3),
                    "f1_score": round(random.uniform(0.75, 0.95), 3)
                },
                "predictions": [
                    {"date": (datetime.datetime.now() + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), 
                     "value": round(random.uniform(100, 200), 2)} 
                    for i in range(10)
                ]
            }
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Created sample JSON file at {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create sample JSON file at {filepath}: {str(e)}")
        return False

def create_sample_csv_file(filepath, data=None):
    """
    Create a sample CSV file at the specified path.
    
    Args:
        filepath (str): Path to save the CSV file
        data (str or list, optional): Custom data to use, or generate random data if None
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Generate data if not provided
        if data is None:
            # Create sample stock data
            df = generate_stock_price_data("SAMPLE", days=30)
            df = add_technical_indicators(df)
            
            # Convert DataFrame to CSV string
            data = df.to_csv()
        
        # If data is not a string (e.g., a list of lists), convert to CSV format
        if not isinstance(data, str):
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            writer.writerows(data)
            data = output.getvalue()
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data)
        
        logger.info(f"Created sample CSV file at {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to create sample CSV file at {filepath}: {str(e)}")
        return False 