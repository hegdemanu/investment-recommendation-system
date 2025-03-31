#!/usr/bin/env python3
"""
Integrated Stock Analysis Tool
Combines technical analysis, price forecasting, and news sentiment analysis
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import traceback

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import our modules
    from trading_engine.analyze_stock_sentiment import SentimentAnalyzer, fetch_news
    from trading_engine.get_stock_data import download_stock_data, calculate_technical_indicators
    
    # Check for Prophet
    from prophet import Prophet
    logger.info("Successfully imported all required modules")
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please install required packages: pip install yfinance pandas numpy matplotlib prophet transformers torch")
    sys.exit(1)

def prepare_prophet_data(df):
    """
    Prepare stock data for Prophet forecasting
    
    Args:
        df (pandas.DataFrame): Stock data with Date and Close columns
        
    Returns:
        pandas.DataFrame: Data formatted for Prophet
    """
    # Create a copy to avoid modifying the original
    prophet_df = df[['Date', 'Close']].copy()
    
    # Rename columns to Prophet's required names
    prophet_df.columns = ['ds', 'y']
    
    return prophet_df

def run_prophet_forecast(data, periods=30, changepoint_prior_scale=0.05):
    """
    Run Prophet forecast on stock data
    
    Args:
        data (pandas.DataFrame): Data formatted for Prophet (ds, y columns)
        periods (int): Number of days to forecast
        changepoint_prior_scale (float): Flexibility of the trend
        
    Returns:
        tuple: (prophet model, forecast dataframe)
    """
    # Create and fit Prophet model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=0.95
    )
    
    # Add stock market seasonality (closed on weekends)
    model.add_country_holidays(country_name='US')
    
    # Fit the model
    model.fit(data)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    
    # Generate forecast
    forecast = model.predict(future)
    
    return model, forecast

def visualize_forecast(model, forecast, historical_data, ticker, output_dir='results'):
    """
    Visualize Prophet forecast
    
    Args:
        model (Prophet): Fitted Prophet model
        forecast (pandas.DataFrame): Forecast dataframe
        historical_data (pandas.DataFrame): Original stock data
        ticker (str): Stock ticker symbol
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 1. Standard Prophet plot
    fig1 = model.plot(forecast)
    ax1 = fig1.get_axes()[0]
    ax1.set_title(f'{ticker} Price Forecast')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    plt.tight_layout()
    plt.savefig(output_path / f"{ticker}_prophet_forecast.png")
    plt.close(fig1)
    
    # 2. Prophet components plot
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig(output_path / f"{ticker}_prophet_components.png")
    plt.close(fig2)
    
    # 3. Custom recent history + forecast plot
    plt.figure(figsize=(12, 6))
    
    # Get the last 90 days of historical data for plotting
    recent_data = historical_data.copy()
    recent_data = recent_data.tail(90)
    
    # Get the forecast for the prediction period
    forecast_tail = forecast[forecast['ds'] > recent_data['Date'].max()]
    
    # Plot historical data
    plt.plot(recent_data['Date'], recent_data['Close'], label='Historical Price', color='blue')
    
    # Plot forecast
    plt.plot(forecast_tail['ds'], forecast_tail['yhat'], label='Forecast', color='red')
    
    # Plot confidence intervals
    plt.fill_between(
        forecast_tail['ds'],
        forecast_tail['yhat_lower'],
        forecast_tail['yhat_upper'],
        color='red',
        alpha=0.2,
        label='95% Confidence Interval'
    )
    
    plt.title(f'{ticker} - 90-Day History and {len(forecast_tail)}-Day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{ticker}_recent_forecast.png")
    plt.close()
    
    return

def analyze_technical_indicators(df, ticker, output_dir='results'):
    """
    Analyze and visualize technical indicators
    
    Args:
        df (pandas.DataFrame): Stock data with technical indicators
        ticker (str): Stock ticker symbol
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Get recent data (last 120 days)
    recent_df = df.tail(120).copy()
    
    # 1. Price with Moving Averages
    plt.figure(figsize=(12, 6))
    plt.plot(recent_df['Date'], recent_df['Close'], label='Close Price')
    plt.plot(recent_df['Date'], recent_df['MA20'], label='20-day MA', alpha=0.7)
    plt.plot(recent_df['Date'], recent_df['MA50'], label='50-day MA', alpha=0.7)
    plt.plot(recent_df['Date'], recent_df['MA200'], label='200-day MA', alpha=0.7)
    plt.title(f'{ticker} - Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{ticker}_moving_averages.png")
    plt.close()
    
    # 2. MACD
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(recent_df['Date'], recent_df['Close'], label='Close Price')
    plt.title(f'{ticker} - Price and MACD')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(recent_df['Date'], recent_df['MACD'], label='MACD')
    plt.plot(recent_df['Date'], recent_df['MACD_Signal'], label='Signal Line')
    plt.bar(recent_df['Date'], recent_df['MACD_Hist'], label='Histogram', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{ticker}_macd.png")
    plt.close()
    
    # 3. RSI
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(recent_df['Date'], recent_df['Close'], label='Close Price')
    plt.title(f'{ticker} - Price and RSI')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(recent_df['Date'], recent_df['RSI'], label='RSI')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{ticker}_rsi.png")
    plt.close()
    
    # 4. Bollinger Bands
    plt.figure(figsize=(12, 6))
    plt.plot(recent_df['Date'], recent_df['Close'], label='Close Price')
    plt.plot(recent_df['Date'], recent_df['BB_Middle'], label='20-day MA', color='orange')
    plt.plot(recent_df['Date'], recent_df['BB_Upper'], label='Upper Band', color='red')
    plt.plot(recent_df['Date'], recent_df['BB_Lower'], label='Lower Band', color='green')
    plt.title(f'{ticker} - Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"{ticker}_bollinger.png")
    plt.close()
    
    return

def generate_technical_signals(df):
    """
    Generate trading signals based on technical indicators
    
    Args:
        df (pandas.DataFrame): Stock data with technical indicators
        
    Returns:
        dict: Trading signals and their explanations
    """
    # Get the most recent data point
    latest = df.iloc[-1]
    
    signals = {
        "ma_signal": None,
        "macd_signal": None,
        "rsi_signal": None,
        "bollinger_signal": None,
        "explanations": {}
    }
    
    # MA Signal (comparing current price with MAs)
    ma_conditions = []
    
    if latest['Close'] > latest['MA20']:
        ma_conditions.append("Price above 20-day MA")
    else:
        ma_conditions.append("Price below 20-day MA")
        
    if latest['Close'] > latest['MA50']:
        ma_conditions.append("Price above 50-day MA")
    else:
        ma_conditions.append("Price below 50-day MA")
        
    if latest['Close'] > latest['MA200']:
        ma_conditions.append("Price above 200-day MA")
    else:
        ma_conditions.append("Price below 200-day MA")
    
    # Check if MAs are aligned for strong trend
    if latest['MA20'] > latest['MA50'] > latest['MA200'] and latest['Close'] > latest['MA20']:
        signals["ma_signal"] = "STRONG BUY"
    elif latest['MA20'] < latest['MA50'] < latest['MA200'] and latest['Close'] < latest['MA20']:
        signals["ma_signal"] = "STRONG SELL"
    elif latest['Close'] > latest['MA50']:
        signals["ma_signal"] = "BUY"
    elif latest['Close'] < latest['MA50']:
        signals["ma_signal"] = "SELL"
    else:
        signals["ma_signal"] = "NEUTRAL"
    
    signals["explanations"]["ma"] = ", ".join(ma_conditions)
    
    # MACD Signal
    if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > 0:
        signals["macd_signal"] = "BUY"
        signals["explanations"]["macd"] = "MACD above signal line with positive histogram"
    elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < 0:
        signals["macd_signal"] = "SELL"
        signals["explanations"]["macd"] = "MACD below signal line with negative histogram"
    else:
        signals["macd_signal"] = "NEUTRAL"
        signals["explanations"]["macd"] = "MACD and signal line are not clearly aligned"
    
    # RSI Signal
    if latest['RSI'] > 70:
        signals["rsi_signal"] = "OVERBOUGHT"
        signals["explanations"]["rsi"] = f"RSI is overbought at {latest['RSI']:.2f}"
    elif latest['RSI'] < 30:
        signals["rsi_signal"] = "OVERSOLD"
        signals["explanations"]["rsi"] = f"RSI is oversold at {latest['RSI']:.2f}"
    else:
        signals["rsi_signal"] = "NEUTRAL"
        signals["explanations"]["rsi"] = f"RSI is neutral at {latest['RSI']:.2f}"
    
    # Bollinger Bands Signal
    if latest['Close'] > latest['BB_Upper']:
        signals["bollinger_signal"] = "OVERBOUGHT"
        signals["explanations"]["bollinger"] = "Price is above upper Bollinger Band"
    elif latest['Close'] < latest['BB_Lower']:
        signals["bollinger_signal"] = "OVERSOLD"
        signals["explanations"]["bollinger"] = "Price is below lower Bollinger Band"
    else:
        distance_to_upper = (latest['BB_Upper'] - latest['Close']) / (latest['BB_Upper'] - latest['BB_Lower'])
        if distance_to_upper < 0.2:
            signals["bollinger_signal"] = "APPROACHING OVERBOUGHT"
            signals["explanations"]["bollinger"] = "Price is approaching upper Bollinger Band"
        elif distance_to_upper > 0.8:
            signals["bollinger_signal"] = "APPROACHING OVERSOLD"
            signals["explanations"]["bollinger"] = "Price is approaching lower Bollinger Band"
        else:
            signals["bollinger_signal"] = "NEUTRAL"
            signals["explanations"]["bollinger"] = "Price is within Bollinger Bands"
    
    return signals

def analyze_forecast(forecast, historical_data):
    """
    Analyze Prophet forecast to generate trading signals
    
    Args:
        forecast (pandas.DataFrame): Prophet forecast dataframe
        historical_data (pandas.DataFrame): Original stock data
        
    Returns:
        dict: Forecast analysis and signals
    """
    # Get the latest actual price
    latest_price = historical_data['Close'].iloc[-1]
    
    # Get the forecast data points
    forecast_data = forecast[forecast['ds'] > historical_data['Date'].max()]
    
    # Calculate predicted returns
    forecast_end_price = forecast_data['yhat'].iloc[-1]
    forecast_return = (forecast_end_price - latest_price) / latest_price * 100
    
    # Calculate potential upside (best case) and downside (worst case)
    upside = (forecast_data['yhat_upper'].iloc[-1] - latest_price) / latest_price * 100
    downside = (forecast_data['yhat_lower'].iloc[-1] - latest_price) / latest_price * 100
    
    # Calculate trend strength
    days = len(forecast_data)
    trend_coef = np.polyfit(range(days), forecast_data['yhat'].values, 1)[0]
    trend_strength = trend_coef / latest_price * 100  # Normalized trend coefficient
    
    # Calculate forecast confidence
    confidence_width = (forecast_data['yhat_upper'] - forecast_data['yhat_lower']).mean()
    confidence_ratio = confidence_width / latest_price * 100
    
    # Generate forecast signal
    if forecast_return > 10:
        forecast_signal = "STRONG BUY"
    elif forecast_return > 5:
        forecast_signal = "BUY"
    elif forecast_return > -5:
        forecast_signal = "HOLD"
    elif forecast_return > -10:
        forecast_signal = "SELL"
    else:
        forecast_signal = "STRONG SELL"
    
    # Adjust based on confidence
    if confidence_ratio > 25:
        forecast_confidence = "LOW"
        if "STRONG" in forecast_signal:
            forecast_signal = forecast_signal.replace("STRONG ", "")
    elif confidence_ratio > 15:
        forecast_confidence = "MEDIUM"
    else:
        forecast_confidence = "HIGH"
    
    return {
        "forecast_end_price": forecast_end_price,
        "forecast_return": forecast_return,
        "upside": upside,
        "downside": downside,
        "trend_strength": trend_strength,
        "confidence_ratio": confidence_ratio,
        "forecast_signal": forecast_signal,
        "forecast_confidence": forecast_confidence,
        "forecast_period_days": days
    }

def combine_signals(technical_signals, forecast_analysis, sentiment_score=None):
    """
    Combine different trading signals into a final recommendation
    
    Args:
        technical_signals (dict): Technical analysis signals
        forecast_analysis (dict): Forecast analysis results
        sentiment_score (float, optional): News sentiment score
        
    Returns:
        dict: Final recommendation and supporting data
    """
    # Assign weights to different signals
    weights = {
        "ma": 0.20,
        "macd": 0.15,
        "rsi": 0.15,
        "bollinger": 0.10,
        "forecast": 0.25,
        "sentiment": 0.15
    }
    
    # Adjust weights if sentiment is not available
    if sentiment_score is None:
        weights = {k: v / (1 - weights["sentiment"]) for k, v in weights.items() if k != "sentiment"}
        weights["forecast"] += 0.05  # Boost forecast weight
        weights["ma"] += 0.05  # Boost MA weight
        weights["macd"] += 0.05  # Boost MACD weight
    
    # Convert signals to numeric scores
    signal_scores = {}
    
    # MA signal
    ma_score_map = {"STRONG BUY": 1.0, "BUY": 0.5, "NEUTRAL": 0, "SELL": -0.5, "STRONG SELL": -1.0}
    signal_scores["ma"] = ma_score_map.get(technical_signals["ma_signal"], 0)
    
    # MACD signal
    macd_score_map = {"BUY": 0.8, "NEUTRAL": 0, "SELL": -0.8}
    signal_scores["macd"] = macd_score_map.get(technical_signals["macd_signal"], 0)
    
    # RSI signal
    rsi_score_map = {"OVERBOUGHT": -0.7, "OVERSOLD": 0.7, "NEUTRAL": 0}
    signal_scores["rsi"] = rsi_score_map.get(technical_signals["rsi_signal"], 0)
    
    # Bollinger signal
    bollinger_score_map = {
        "OVERBOUGHT": -0.7, 
        "OVERSOLD": 0.7, 
        "APPROACHING OVERBOUGHT": -0.3,
        "APPROACHING OVERSOLD": 0.3,
        "NEUTRAL": 0
    }
    signal_scores["bollinger"] = bollinger_score_map.get(technical_signals["bollinger_signal"], 0)
    
    # Forecast signal
    forecast_score_map = {
        "STRONG BUY": 1.0, 
        "BUY": 0.6, 
        "HOLD": 0, 
        "SELL": -0.6, 
        "STRONG SELL": -1.0
    }
    signal_scores["forecast"] = forecast_score_map.get(forecast_analysis["forecast_signal"], 0)
    
    # Adjust forecast score based on confidence
    confidence_adj = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}
    forecast_conf_adj = confidence_adj.get(forecast_analysis["forecast_confidence"], 0.5)
    signal_scores["forecast"] *= forecast_conf_adj
    
    # Sentiment score (normalized to -1 to 1 range)
    if sentiment_score is not None:
        signal_scores["sentiment"] = max(min(sentiment_score, 1.0), -1.0)
    
    # Calculate weighted score
    weighted_score = sum(signal_scores[signal] * weights[signal] for signal in signal_scores)
    
    # Generate final recommendation
    if weighted_score > 0.6:
        recommendation = "STRONG BUY"
    elif weighted_score > 0.2:
        recommendation = "BUY"
    elif weighted_score > -0.2:
        recommendation = "HOLD"
    elif weighted_score > -0.6:
        recommendation = "SELL"
    else:
        recommendation = "STRONG SELL"
    
    # Generate confidence level
    score_abs = abs(weighted_score)
    if score_abs > 0.7:
        confidence = "HIGH"
    elif score_abs > 0.4:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "weighted_score": weighted_score,
        "signal_scores": signal_scores,
        "signal_weights": weights
    }

def save_analysis_report(ticker, data, technical_signals, forecast_analysis, 
                         final_recommendation, sentiment_data=None, output_dir="results"):
    """
    Save a comprehensive analysis report
    
    Args:
        ticker (str): Stock ticker symbol
        data (pandas.DataFrame): Stock data
        technical_signals (dict): Technical analysis signals
        forecast_analysis (dict): Forecast analysis results
        final_recommendation (dict): Final recommendation
        sentiment_data (pandas.DataFrame, optional): Sentiment analysis data
        output_dir (str): Directory to save reports
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create report file
    report_path = output_path / f"{ticker}_analysis_report.txt"
    
    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Generate report content
    with open(report_path, "w") as f:
        # Header
        f.write(f"Investment Analysis Report - {ticker}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Date: {current_date}\n")
        company_name = data['CompanyName'].iloc[0] if 'CompanyName' in data.columns else ticker
        f.write(f"Company: {company_name}\n")
        
        if 'Sector' in data.columns:
            f.write(f"Sector: {data['Sector'].iloc[0]}\n")
        if 'Industry' in data.columns:
            f.write(f"Industry: {data['Industry'].iloc[0]}\n")
        
        latest_price = data['Close'].iloc[-1]
        f.write(f"Current Price: ${latest_price:.2f}\n\n")
        
        # Trading Recommendation
        f.write("TRADING RECOMMENDATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Recommendation: {final_recommendation['recommendation']}\n")
        f.write(f"Confidence: {final_recommendation['confidence']}\n")
        f.write(f"Overall Score: {final_recommendation['weighted_score']:.2f}\n\n")
        
        # Technical Analysis
        f.write("TECHNICAL ANALYSIS\n")
        f.write("-" * 30 + "\n")
        
        f.write(f"Moving Averages: {technical_signals['ma_signal']}\n")
        f.write(f"  {technical_signals['explanations']['ma']}\n\n")
        
        f.write(f"MACD: {technical_signals['macd_signal']}\n")
        f.write(f"  {technical_signals['explanations']['macd']}\n\n")
        
        f.write(f"RSI: {technical_signals['rsi_signal']}\n")
        f.write(f"  {technical_signals['explanations']['rsi']}\n\n")
        
        f.write(f"Bollinger Bands: {technical_signals['bollinger_signal']}\n")
        f.write(f"  {technical_signals['explanations']['bollinger']}\n\n")
        
        # Price Forecast
        f.write("PRICE FORECAST\n")
        f.write("-" * 30 + "\n")
        
        f.write(f"Forecast Signal: {forecast_analysis['forecast_signal']}\n")
        f.write(f"Forecast Confidence: {forecast_analysis['forecast_confidence']}\n")
        f.write(f"Forecast Period: {forecast_analysis['forecast_period_days']} days\n")
        f.write(f"Forecasted Price: ${forecast_analysis['forecast_end_price']:.2f}\n")
        f.write(f"Expected Return: {forecast_analysis['forecast_return']:.2f}%\n")
        f.write(f"Potential Upside: {forecast_analysis['upside']:.2f}%\n")
        f.write(f"Potential Downside: {forecast_analysis['downside']:.2f}%\n\n")
        
        # Sentiment Analysis (if available)
        if sentiment_data is not None:
            f.write("NEWS SENTIMENT ANALYSIS\n")
            f.write("-" * 30 + "\n")
            
            avg_sentiment = sentiment_data['sentiment_score'].mean()
            f.write(f"Average Sentiment Score: {avg_sentiment:.4f}\n")
            
            if avg_sentiment > 0.3:
                sentiment_desc = "VERY POSITIVE"
            elif avg_sentiment > 0.1:
                sentiment_desc = "POSITIVE"
            elif avg_sentiment > -0.1:
                sentiment_desc = "NEUTRAL"
            elif avg_sentiment > -0.3:
                sentiment_desc = "NEGATIVE"
            else:
                sentiment_desc = "VERY NEGATIVE"
            
            f.write(f"Overall Market Sentiment: {sentiment_desc}\n")
            
            # Count articles by sentiment
            sentiment_counts = sentiment_data['sentiment_label'].value_counts()
            f.write(f"Article Distribution: ")
            for sentiment, count in sentiment_counts.items():
                f.write(f"{sentiment.capitalize()}: {count} ({count/len(sentiment_data)*100:.1f}%), ")
            f.write("\n\n")
            
            # Most positive and negative headlines
            f.write("Top Positive News:\n")
            top_positive = sentiment_data.sort_values('sentiment_score', ascending=False).head(2)
            for i, (_, article) in enumerate(top_positive.iterrows(), 1):
                f.write(f"  {i}. {article['title']} (Score: {article['sentiment_score']:.2f})\n")
            
            f.write("\nTop Negative News:\n")
            top_negative = sentiment_data.sort_values('sentiment_score').head(2)
            for i, (_, article) in enumerate(top_negative.iterrows(), 1):
                f.write(f"  {i}. {article['title']} (Score: {article['sentiment_score']:.2f})\n")
            f.write("\n")
        
        # Signal Contributions
        f.write("SIGNAL CONTRIBUTIONS\n")
        f.write("-" * 30 + "\n")
        
        for signal, score in final_recommendation['signal_scores'].items():
            weight = final_recommendation['signal_weights'][signal]
            contribution = score * weight
            f.write(f"{signal.upper()}: Score {score:.2f} × Weight {weight:.2f} = {contribution:.2f}\n")
        
        f.write("\nDisclaimer: This analysis is for informational purposes only and should not be considered financial advice.\n")
        f.write("Past performance is not indicative of future results. Always consult a financial advisor before making investment decisions.\n")
    
    logger.info(f"Analysis report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Integrated Stock Analysis Tool')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--years', type=int, default=5, help='Years of historical data to analyze')
    parser.add_argument('--days', type=int, default=30, help='Days to forecast')
    parser.add_argument('--api_key', type=str, help='NewsAPI API key for sentiment analysis')
    parser.add_argument('--news_days', type=int, default=7, help='Days of news to analyze')
    parser.add_argument('--skip_sentiment', action='store_true', help='Skip sentiment analysis')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    ticker = args.ticker
    forecast_days = args.days
    history_years = args.years
    output_dir = args.output
    
    logger.info(f"Starting integrated analysis for {ticker}")
    
    try:
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        
        # 1. Download historical stock data
        logger.info(f"Downloading {history_years} years of historical data for {ticker}")
        stock_data = download_stock_data(ticker, years=history_years)
        
        if stock_data is None or stock_data.empty:
            logger.error(f"Failed to download data for {ticker}")
            return
        
        # 2. Calculate technical indicators
        logger.info("Calculating technical indicators")
        stock_data_with_indicators = calculate_technical_indicators(stock_data)
        
        # 3. Analyze technical indicators
        logger.info("Analyzing technical indicators")
        analyze_technical_indicators(stock_data_with_indicators, ticker, output_dir=output_dir)
        technical_signals = generate_technical_signals(stock_data_with_indicators)
        
        # 4. Run price forecast
        logger.info(f"Running price forecast for next {forecast_days} days")
        prophet_data = prepare_prophet_data(stock_data)
        model, forecast = run_prophet_forecast(prophet_data, periods=forecast_days)
        visualize_forecast(model, forecast, stock_data, ticker, output_dir=output_dir)
        forecast_analysis = analyze_forecast(forecast, stock_data)
        
        # 5. Run sentiment analysis (optional)
        sentiment_data = None
        if not args.skip_sentiment:
            # Get API key
            api_key = args.api_key
            if not api_key:
                api_key = os.environ.get('NEWSAPI_KEY')
            
            if api_key:
                logger.info(f"Running sentiment analysis for {ticker} news")
                try:
                    # Fetch news
                    news_articles = fetch_news(api_key, ticker, days=args.news_days)
                    
                    if news_articles:
                        # Initialize sentiment analyzer
                        analyzer = SentimentAnalyzer()
                        
                        # Analyze sentiment
                        results = analyzer.analyze_news_batch(news_articles)
                        
                        if results:
                            # Convert to DataFrame
                            sentiment_data = pd.DataFrame(results)
                            sentiment_data['published_at'] = pd.to_datetime(sentiment_data['published_at'])
                            
                            # Plot sentiment visualizations
                            from trading_engine.analyze_stock_sentiment import visualize_sentiment, save_sentiment_report
                            visualize_sentiment(results, ticker)
                            save_sentiment_report(sentiment_data, ticker)
                except Exception as e:
                    logger.error(f"Error in sentiment analysis: {e}")
                    logger.error(traceback.format_exc())
                    sentiment_data = None
            else:
                logger.warning("Skipping sentiment analysis - No NewsAPI key provided")
        
        # 6. Combine signals and generate final recommendation
        sentiment_score = None
        if sentiment_data is not None and not sentiment_data.empty:
            sentiment_score = sentiment_data['sentiment_score'].mean()
            logger.info(f"Average sentiment score: {sentiment_score:.4f}")
        
        logger.info("Generating final recommendation")
        final_recommendation = combine_signals(technical_signals, forecast_analysis, sentiment_score)
        
        # 7. Save analysis report
        save_analysis_report(
            ticker, 
            stock_data, 
            technical_signals, 
            forecast_analysis, 
            final_recommendation,
            sentiment_data,
            output_dir=output_dir
        )
        
        # 8. Print summary
        logger.info(f"\nAnalysis complete for {ticker}")
        logger.info(f"Recommendation: {final_recommendation['recommendation']} (Confidence: {final_recommendation['confidence']})")
        logger.info(f"Technical Analysis: MA={technical_signals['ma_signal']}, MACD={technical_signals['macd_signal']}, RSI={technical_signals['rsi_signal']}")
        logger.info(f"Price Forecast: {forecast_analysis['forecast_signal']} (Expected Return: {forecast_analysis['forecast_return']:.2f}%)")
        
        if sentiment_score is not None:
            if sentiment_score > 0.3:
                sentiment_desc = "VERY POSITIVE"
            elif sentiment_score > 0.1:
                sentiment_desc = "POSITIVE"
            elif sentiment_score > -0.1:
                sentiment_desc = "NEUTRAL"
            elif sentiment_score > -0.3:
                sentiment_desc = "NEGATIVE"
            else:
                sentiment_desc = "VERY NEGATIVE"
            logger.info(f"News Sentiment: {sentiment_desc} (Score: {sentiment_score:.4f})")
        
        logger.info(f"Detailed report and visualizations saved to {output_dir}/ directory")
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 