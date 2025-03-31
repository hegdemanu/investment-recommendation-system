import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
import yfinance as yf
import argparse

# Add the project root to Python path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the models
from trading_engine.models.sentiment_model_implementation import SentimentAnalyzer
from trading_engine.models.prophet_model_implementation import ProphetForecaster

def fix_yfinance_data(data):
    """
    Fix the data format from yfinance to work with Prophet
    
    Args:
        data: DataFrame from yfinance
        
    Returns:
        Fixed DataFrame
    """
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Fix any tuple column names from yfinance
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    # Ensure column names are strings
    data.columns = [str(col) for col in data.columns]
    
    return data

def get_trading_signal(sentiment, forecast_change):
    """
    Generate a trading signal based on sentiment and forecast
    
    Args:
        sentiment: Dictionary with sentiment analysis results
        forecast_change: Forecasted price change percentage
        
    Returns:
        Dictionary with trading recommendation
    """
    # Convert sentiment to numeric score (-1 to 1)
    sentiment_scores = {
        'positive': 1.0,
        'negative': -1.0,
        'neutral': 0.0
    }
    
    sentiment_score = sentiment_scores.get(sentiment['overall_sentiment'], 0.0) * sentiment['confidence']
    
    # Define thresholds
    sentiment_threshold = 0.3
    forecast_threshold = 2.0  # 2% price change
    
    # Generate signal based on sentiment and forecast
    if sentiment_score > sentiment_threshold and forecast_change > forecast_threshold:
        signal = "STRONG BUY"
        reasoning = "Positive sentiment and bullish price forecast"
        confidence = (sentiment['confidence'] + min(forecast_change / 10, 1.0)) / 2
    elif sentiment_score > sentiment_threshold and forecast_change > 0:
        signal = "BUY"
        reasoning = "Positive sentiment with slight price increase forecast"
        confidence = sentiment['confidence'] * 0.8
    elif sentiment_score < -sentiment_threshold and forecast_change < -forecast_threshold:
        signal = "STRONG SELL"
        reasoning = "Negative sentiment and bearish price forecast"
        confidence = (sentiment['confidence'] + min(abs(forecast_change) / 10, 1.0)) / 2
    elif sentiment_score < -sentiment_threshold and forecast_change < 0:
        signal = "SELL"
        reasoning = "Negative sentiment with slight price decrease forecast"
        confidence = sentiment['confidence'] * 0.8
    elif forecast_change > forecast_threshold:
        signal = "BUY"
        reasoning = "Bullish price forecast despite neutral sentiment"
        confidence = min(forecast_change / 10, 1.0) * 0.7
    elif forecast_change < -forecast_threshold:
        signal = "SELL"
        reasoning = "Bearish price forecast despite neutral sentiment"
        confidence = min(abs(forecast_change) / 10, 1.0) * 0.7
    else:
        signal = "HOLD"
        reasoning = "No strong signals in sentiment or price forecast"
        confidence = 0.5
        
    return {
        "signal": signal,
        "reasoning": reasoning,
        "confidence": confidence,
        "sentiment_score": sentiment_score,
        "forecast_change": forecast_change
    }

def main():
    """
    Script to combine sentiment analysis and price forecasting for trading decisions
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Combine sentiment and forecasting for trading decisions')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to forecast')
    parser.add_argument('--years', type=int, default=2, help='Years of historical data to use')
    parser.add_argument('--api_key', type=str, required=True, help='News API key')
    parser.add_argument('--news_days', type=int, default=7, help='Days of news to analyze')
    args = parser.parse_args()
    
    ticker = args.ticker
    forecast_days = args.days
    years_data = args.years
    api_key = args.api_key
    news_days = args.news_days
    
    print("=" * 80)
    print(f"TRADING RECOMMENDATION SYSTEM FOR {ticker}")
    print("=" * 80)
    
    # Create output directory
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Download historical stock data
    print(f"\nDownloading {years_data} years of historical data for {ticker}...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_data)
    
    try:
        stock_data = yf.download(
            ticker, 
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        # Check if data is valid
        if len(stock_data) < 30:
            print(f"Error: Not enough data points for {ticker}. Received only {len(stock_data)} days of data.")
            return
            
        print(f"Successfully downloaded {len(stock_data)} days of historical data.")
        
        # Fix the data format
        data = fix_yfinance_data(stock_data)
        
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return
    
    # Step 2: Initialize models
    print("\nInitializing models...")
    sentiment_analyzer = SentimentAnalyzer()
    forecaster = ProphetForecaster()
    
    # Step 3: Analyze sentiment
    print(f"\nAnalyzing news sentiment for {ticker} (past {news_days} days)...")
    sentiment = sentiment_analyzer.analyze_news(ticker, days=news_days, api_key=api_key)
    
    if sentiment['article_count'] == 0:
        print(f"Warning: No news articles found for {ticker} in the past {news_days} days.")
        print("Proceeding with neutral sentiment.")
    else:
        print(f"Found {sentiment['article_count']} news articles.")
        print(f"Overall sentiment: {sentiment['overall_sentiment'].upper()} (confidence: {sentiment['confidence']:.4f})")
        
    # Step 4: Train forecast model
    print(f"\nTraining price forecast model for {ticker}...")
    
    # Split data for training
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # Fine-tune parameters for financial data
    model_params = {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10.0,
        "changepoint_range": 0.9,
        "daily_seasonality": False,
        "weekly_seasonality": True,
        "yearly_seasonality": True
    }
    
    train_result = forecaster.train(train_data, target_col='Close', params=model_params)
    
    if not train_result['success']:
        print(f"Error training model: {train_result.get('error', 'Unknown error')}")
        return
        
    print("Forecast model trained successfully.")
    
    # Step 5: Generate forecast
    print(f"\nGenerating {forecast_days}-day price forecast...")
    forecast = forecaster.predict(periods=forecast_days)
    
    # Calculate expected price movement
    last_close = data['Close'].iloc[-1]
    forecasted_price = forecast['yhat'].iloc[-1]
    price_change = ((forecasted_price - last_close) / last_close) * 100
    
    print(f"Last Close Price: ${last_close:.2f}")
    print(f"Forecasted Price ({forecast_days} days): ${forecasted_price:.2f}")
    print(f"Expected Movement: {price_change:.2f}%")
    
    # Step 6: Generate trading recommendation
    print("\nGenerating trading recommendation...")
    signal = get_trading_signal(sentiment, price_change)
    
    print("\n" + "=" * 80)
    print(f"TRADING RECOMMENDATION FOR {ticker}")
    print("=" * 80)
    print(f"Signal: {signal['signal']}")
    print(f"Confidence: {signal['confidence']:.2f}")
    print(f"Reasoning: {signal['reasoning']}")
    print("\nBased on:")
    print(f"  - Sentiment: {sentiment['overall_sentiment'].upper()} (confidence: {sentiment['confidence']:.2f})")
    print(f"  - {forecast_days}-day Price Forecast: {price_change:.2f}%")
    
    # Step 7: Create visualization
    print("\nCreating visualization of combined analysis...")
    
    # Create combined visualization
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Price history and forecast
    plt.subplot(2, 1, 1)
    
    # Plot historical data
    plt.plot(data['Date'], data['Close'], 'b-', label='Historical Price')
    
    # Get future dates beyond the historical data
    last_date = data['Date'].max()
    future_forecast = forecast[forecast['ds'] > pd.to_datetime(last_date)]
    
    # Plot the forecast
    plt.plot(future_forecast['ds'], future_forecast['yhat'], 'r-', label='Forecast')
    plt.fill_between(
        future_forecast['ds'], 
        future_forecast['yhat_lower'], 
        future_forecast['yhat_upper'],
        color='red', alpha=0.2, label='Prediction Interval'
    )
    
    # Format the plot
    plt.title(f'{ticker} Stock Price History and {forecast_days}-day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Subplot 2: Sentiment distribution
    plt.subplot(2, 1, 2)
    
    # Create pie chart of sentiment distribution
    if sentiment['article_count'] > 0:
        sentiment_dist = sentiment['sentiment_distribution']
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [sentiment_dist['positive'], sentiment_dist['negative'], sentiment_dist['neutral']]
        colors = ['green', 'red', 'gray']
        explode = (0.1, 0.1, 0.1)  # explode all slices
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.title(f'News Sentiment Distribution for {ticker} (Past {news_days} Days)')
    else:
        plt.text(0.5, 0.5, f"No news articles found for {ticker} in the past {news_days} days", 
                 ha='center', va='center', fontsize=12)
    
    # Add the trading recommendation
    plt.figtext(0.5, 0.02, f"RECOMMENDATION: {signal['signal']} ({signal['confidence']:.2f} confidence)", 
                ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the visualization
    viz_path = os.path.join(results_dir, f'{ticker}_trading_recommendation.png')
    plt.savefig(viz_path)
    print(f"Trading recommendation visualization saved to {viz_path}")
    
    # Save the trading recommendation to a text file
    recommendation_path = os.path.join(results_dir, f'{ticker}_recommendation.txt')
    with open(recommendation_path, 'w') as f:
        f.write(f"TRADING RECOMMENDATION FOR {ticker}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"Signal: {signal['signal']}\n")
        f.write(f"Confidence: {signal['confidence']:.2f}\n")
        f.write(f"Reasoning: {signal['reasoning']}\n\n")
        f.write("Based on:\n")
        f.write(f"  - Sentiment: {sentiment['overall_sentiment'].upper()} (confidence: {sentiment['confidence']:.2f})\n")
        f.write(f"  - {forecast_days}-day Price Forecast: {price_change:.2f}%\n\n")
        f.write("Market Data:\n")
        f.write(f"  - Last Close Price: ${last_close:.2f}\n")
        f.write(f"  - Forecasted Price ({forecast_days} days): ${forecasted_price:.2f}\n")
        
        if sentiment['article_count'] > 0:
            f.write("\nRecent News Analysis:\n")
            for i, article in enumerate(sentiment['articles'][:5]):  # Top 5 articles
                f.write(f"  {i+1}. {article['title']}\n")
                f.write(f"     Sentiment: {article['sentiment']['sentiment'].upper()} (confidence: {article['sentiment']['confidence']:.2f})\n")
    
    print(f"Trading recommendation details saved to {recommendation_path}")
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main() 