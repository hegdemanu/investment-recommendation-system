import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta
import argparse

# Add the project root to Python path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the sentiment analyzer
from trading_engine.models.sentiment_model_implementation import SentimentAnalyzer

def main():
    """
    Script to analyze real news sentiment using the pre-trained FinBERT model
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze real news sentiment for a stock ticker')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol to analyze (e.g., AAPL)')
    parser.add_argument('--days', type=int, default=7, help='Number of days of news to analyze')
    parser.add_argument('--api_key', type=str, required=True, help='News API key')
    args = parser.parse_args()
    
    ticker = args.ticker
    days = args.days
    api_key = args.api_key
    
    print("=" * 80)
    print(f"REAL NEWS SENTIMENT ANALYSIS FOR {ticker}")
    print("=" * 80)
    
    # Initialize the sentiment analyzer
    print("\nInitializing sentiment analyzer with pre-trained FinBERT model...")
    sentiment_analyzer = SentimentAnalyzer()
    
    # Analyze news with real API
    print(f"\nFetching and analyzing real news for {ticker} (past {days} days)...")
    news_analysis = sentiment_analyzer.analyze_news(ticker, days=days, api_key=api_key)
    
    if news_analysis['article_count'] == 0:
        print(f"No news articles found for {ticker} in the past {days} days.")
        return
    
    # Show sentiment analysis results
    print(f"\nFound {news_analysis['article_count']} news articles for {ticker}")
    print("\nArticle Sentiments:")
    print("-" * 80)
    
    # Display sentiment for each article
    for i, article in enumerate(news_analysis['articles']):
        print(f"\n{i+1}. Date: {article['date']}")
        print(f"   Title: {article['title']}")
        print(f"   URL: {article['url']}")
        print(f"   Sentiment: {article['sentiment']['sentiment'].upper()}")
        print(f"   Confidence: {article['sentiment']['confidence']:.4f}")
    
    # Display overall sentiment
    print("\n" + "=" * 80)
    print(f"OVERALL SENTIMENT FOR {ticker}:")
    print("-" * 80)
    print(f"Sentiment: {news_analysis['overall_sentiment'].upper()}")
    print(f"Confidence: {news_analysis['confidence']:.4f}")
    print(f"Sentiment Distribution:")
    print(f"   Positive: {news_analysis['sentiment_distribution']['positive']:.4f}")
    print(f"   Negative: {news_analysis['sentiment_distribution']['negative']:.4f}")
    print(f"   Neutral: {news_analysis['sentiment_distribution']['neutral']:.4f}")
    
    # Create visualizations
    print("\nCreating sentiment visualizations...")
    
    # Create a pie chart for sentiment distribution
    plt.figure(figsize=(10, 6))
    
    # Create pie chart of sentiment distribution
    sentiment_dist = news_analysis['sentiment_distribution']
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [sentiment_dist['positive'], sentiment_dist['negative'], sentiment_dist['neutral']]
    colors = ['green', 'red', 'gray']
    explode = (0.1, 0.1, 0.1)  # explode all slices
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
            autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title(f'Sentiment Distribution for {ticker} News')
    
    # Save the chart
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{ticker}_sentiment_distribution.png')
    print(f"\nSentiment distribution chart saved to results/{ticker}_sentiment_distribution.png")
    
    # Create a timeline of article sentiments
    plt.figure(figsize=(12, 6))
    
    # Extract data for timeline
    dates = [datetime.strptime(a['date'].split('T')[0], '%Y-%m-%d') 
             if 'T' in a['date'] else datetime.strptime(a['date'], '%Y-%m-%d') 
             for a in news_analysis['articles']]
    
    sentiments = [a['sentiment']['sentiment'] for a in news_analysis['articles']]
    confidences = [a['sentiment']['confidence'] for a in news_analysis['articles']]
    
    # Sort by date
    date_sentiment = list(zip(dates, sentiments, confidences))
    date_sentiment.sort(key=lambda x: x[0])
    dates, sentiments, confidences = zip(*date_sentiment)
    
    # Create color mapping
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
    bar_colors = [colors[s] for s in sentiments]
    
    # Create bar chart
    plt.bar(range(len(dates)), confidences, color=bar_colors)
    plt.xticks(range(len(dates)), [d.strftime('%Y-%m-%d') for d in dates], rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Confidence Score')
    plt.title(f'Sentiment Timeline for {ticker} News')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[s], label=s.capitalize()) for s in set(sentiments)]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    
    # Save the chart
    plt.savefig(f'results/{ticker}_sentiment_timeline.png')
    print(f"Sentiment timeline chart saved to results/{ticker}_sentiment_timeline.png")
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main() 