#!/usr/bin/env python3
"""
Stock Sentiment Analysis Tool
Uses FinBERT to analyze sentiment from news articles for a specific stock
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import argparse
import logging
from pathlib import Path
import requests
import json
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the transformers library for FinBERT
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    logger.info("Successfully imported transformers and torch")
except ImportError:
    logger.error("transformers or torch not installed. Install using: pip install transformers torch")
    sys.exit(1)

class SentimentAnalyzer:
    """Class for analyzing sentiment using FinBERT"""
    
    def __init__(self):
        """Initialize the FinBERT model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise e
    
    def analyze_text(self, text):
        """Analyze the sentiment of a given text"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # FinBERT predicts positive, negative, neutral
        labels = ["positive", "negative", "neutral"]
        scores = {labels[i]: float(predictions[0][i]) for i in range(len(labels))}
        
        # Convert to a single sentiment score (-1 to 1)
        sentiment_score = scores["positive"] - scores["negative"]
        
        return {
            "scores": scores,
            "sentiment_score": sentiment_score,
            "label": labels[torch.argmax(predictions[0]).item()]
        }
    
    def analyze_news_batch(self, news_articles):
        """Analyze a batch of news articles"""
        results = []
        
        for article in news_articles:
            title = article.get("title", "")
            description = article.get("description", "")
            content = article.get("content", "")
            
            # Combine title and description for analysis
            text = f"{title}. {description}"
            if content and len(content) > 10:
                text += f" {content}"
            
            # Skip empty articles
            if len(text.strip()) < 10:
                continue
            
            # Analyze sentiment
            sentiment = self.analyze_text(text)
            
            # Add to results
            results.append({
                "title": title,
                "description": description,
                "published_at": article.get("publishedAt", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "url": article.get("url", ""),
                "sentiment_score": sentiment["sentiment_score"],
                "sentiment_label": sentiment["label"],
                "positive_score": sentiment["scores"]["positive"],
                "negative_score": sentiment["scores"]["negative"],
                "neutral_score": sentiment["scores"]["neutral"]
            })
        
        return results

def fetch_news(api_key, ticker, days=7):
    """Fetch news for a specific ticker using NewsAPI"""
    # Calculate the date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Format dates for API
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Construct the API URL
    url = f"https://newsapi.org/v2/everything"
    
    # Parameters for the API request
    params = {
        'q': ticker,
        'from': from_date,
        'to': to_date,
        'language': 'en',
        'sortBy': 'publishedAt',
        'apiKey': api_key
    }
    
    try:
        # Make the API request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the JSON response
        data = response.json()
        
        # Check if we got articles
        if data.get('status') == 'ok' and data.get('totalResults', 0) > 0:
            return data.get('articles', [])
        else:
            logger.warning(f"No news articles found for {ticker}")
            return []
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching news: {e}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response: {e}")
        return []

def visualize_sentiment(results, ticker):
    """Create visualizations for sentiment analysis"""
    if not results:
        logger.warning("No results to visualize")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    # Convert published_at to datetime
    df['published_at'] = pd.to_datetime(df['published_at'])
    df = df.sort_values('published_at')
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment_label'].value_counts()
    sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
    plt.title(f'Sentiment Distribution for {ticker} News')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    plt.savefig(output_dir / f"{ticker}_sentiment_distribution.png")
    
    # 2. Sentiment Timeline
    plt.figure(figsize=(12, 6))
    plt.plot(df['published_at'], df['sentiment_score'], marker='o', linestyle='-')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.title(f'Sentiment Timeline for {ticker}')
    plt.xlabel('Publication Date')
    plt.ylabel('Sentiment Score (-1 to 1)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{ticker}_sentiment_timeline.png")
    
    # 3. Source Distribution
    plt.figure(figsize=(12, 6))
    source_counts = df['source'].value_counts().head(10)  # Top 10 sources
    source_counts.plot(kind='bar')
    plt.title(f'Top News Sources for {ticker}')
    plt.xlabel('Source')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f"{ticker}_news_sources.png")
    
    logger.info(f"Visualizations saved to results/ directory")
    
    return df

def save_sentiment_report(df, ticker):
    """Save a text report of the sentiment analysis"""
    if df is None or df.empty:
        logger.warning("No data to save in report")
        return
    
    # Create output directory
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Calculate average sentiment
    avg_sentiment = df['sentiment_score'].mean()
    
    # Determine overall sentiment
    if avg_sentiment > 0.3:
        overall = "VERY POSITIVE"
    elif avg_sentiment > 0.1:
        overall = "POSITIVE"
    elif avg_sentiment > -0.1:
        overall = "NEUTRAL"
    elif avg_sentiment > -0.3:
        overall = "NEGATIVE"
    else:
        overall = "VERY NEGATIVE"
    
    # Count articles by sentiment
    sentiment_counts = df['sentiment_label'].value_counts()
    
    # Create the report
    with open(output_dir / f"{ticker}_sentiment_report.txt", 'w') as f:
        f.write(f"Sentiment Analysis Report for {ticker}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Period: {df['published_at'].min().strftime('%Y-%m-%d')} to {df['published_at'].max().strftime('%Y-%m-%d')}\n")
        f.write(f"Total Articles Analyzed: {len(df)}\n\n")
        
        f.write("Sentiment Summary:\n")
        f.write(f"  Average Sentiment Score: {avg_sentiment:.4f}\n")
        f.write(f"  Overall Market Sentiment: {overall}\n\n")
        
        f.write("Article Distribution:\n")
        for sentiment, count in sentiment_counts.items():
            f.write(f"  {sentiment.capitalize()}: {count} articles ({count/len(df)*100:.1f}%)\n")
        
        f.write("\nTop 5 Most Positive Articles:\n")
        top_positive = df.sort_values('sentiment_score', ascending=False).head(5)
        for i, (_, article) in enumerate(top_positive.iterrows(), 1):
            f.write(f"  {i}. {article['title']} (Score: {article['sentiment_score']:.4f})\n")
            f.write(f"     Source: {article['source']} | {article['published_at'].strftime('%Y-%m-%d')}\n")
            f.write(f"     URL: {article['url']}\n\n")
        
        f.write("\nTop 5 Most Negative Articles:\n")
        top_negative = df.sort_values('sentiment_score').head(5)
        for i, (_, article) in enumerate(top_negative.iterrows(), 1):
            f.write(f"  {i}. {article['title']} (Score: {article['sentiment_score']:.4f})\n")
            f.write(f"     Source: {article['source']} | {article['published_at'].strftime('%Y-%m-%d')}\n")
            f.write(f"     URL: {article['url']}\n\n")
    
    logger.info(f"Sentiment report saved to results/{ticker}_sentiment_report.txt")

def main():
    parser = argparse.ArgumentParser(description='Stock Sentiment Analysis Tool')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--days', type=int, default=7, help='Number of days of news to analyze')
    parser.add_argument('--api_key', type=str, help='NewsAPI API key')
    args = parser.parse_args()
    
    ticker = args.ticker
    days = args.days
    
    # Check for API key
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get('NEWSAPI_KEY')
        if not api_key:
            logger.error("No NewsAPI key provided. Please provide one using --api_key or set the NEWSAPI_KEY environment variable.")
            logger.error("You can get a free API key at https://newsapi.org/")
            sys.exit(1)
    
    logger.info(f"Analyzing sentiment for {ticker} over the past {days} days")
    
    # Fetch news
    news_articles = fetch_news(api_key, ticker, days)
    logger.info(f"Found {len(news_articles)} news articles for {ticker}")
    
    if not news_articles:
        logger.warning("No news articles found. Exiting.")
        sys.exit(0)
    
    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze sentiment
    results = analyzer.analyze_news_batch(news_articles)
    logger.info(f"Analyzed sentiment for {len(results)} articles")
    
    # Visualize results
    df = visualize_sentiment(results, ticker)
    
    # Save report
    save_sentiment_report(df, ticker)
    
    # Print summary
    avg_sentiment = df['sentiment_score'].mean()
    logger.info(f"\nSentiment Analysis Summary for {ticker}:")
    logger.info(f"Average Sentiment Score: {avg_sentiment:.4f}")
    
    # Determine trading recommendation based on sentiment
    if avg_sentiment > 0.3:
        recommendation = "STRONG BUY"
    elif avg_sentiment > 0.1:
        recommendation = "BUY"
    elif avg_sentiment > -0.1:
        recommendation = "HOLD"
    elif avg_sentiment > -0.3:
        recommendation = "SELL"
    else:
        recommendation = "STRONG SELL"
    
    logger.info(f"Sentiment-Based Trading Recommendation: {recommendation}")
    logger.info(f"Detailed report saved to results/{ticker}_sentiment_report.txt")

if __name__ == "__main__":
    main() 