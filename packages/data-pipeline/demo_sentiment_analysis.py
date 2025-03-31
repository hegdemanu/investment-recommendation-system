import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

# Add the project root to Python path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the sentiment analyzer
from trading_engine.models.sentiment_model_implementation import SentimentAnalyzer

def main():
    """
    Demo script to test the sentiment analyzer using a pre-trained FinBERT model
    """
    print("=" * 80)
    print("SENTIMENT ANALYSIS DEMO USING PRE-TRAINED FINBERT MODEL")
    print("=" * 80)
    
    # Initialize the sentiment analyzer
    print("\nInitializing sentiment analyzer with pre-trained FinBERT model...")
    sentiment_analyzer = SentimentAnalyzer()
    
    # Test with some financial news headlines
    headlines = [
        "Company XYZ reports record quarterly profits, beating analyst expectations",
        "Market crashes as inflation fears intensify",
        "Federal Reserve announces plans to hold interest rates steady",
        "Company ABC under investigation for accounting irregularities",
        "Technology stocks rally after positive earnings reports"
    ]
    
    print("\nAnalyzing sample financial headlines:")
    print("-" * 80)
    
    # Analyze each headline
    for i, headline in enumerate(headlines):
        print(f"\nHeadline {i+1}: {headline}")
        result = sentiment_analyzer.analyze_text(headline)
        
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Raw scores: Positive={result['raw_scores']['positive']:.4f}, " +
              f"Negative={result['raw_scores']['negative']:.4f}, " +
              f"Neutral={result['raw_scores']['neutral']:.4f}")
    
    # Create a simulated news dataset for a stock
    print("\n" + "=" * 80)
    print("SIMULATED STOCK NEWS ANALYSIS")
    print("=" * 80)
    
    # Create a simulated news dataset for Apple (AAPL)
    ticker = "AAPL"
    
    # Create simulated news data for Apple covering the last 7 days
    today = datetime.now()
    news_data = [
        {
            "title": "Apple's iPhone 15 Sales Exceed Expectations in Asian Markets",
            "description": "Apple reported that iPhone 15 sales have significantly exceeded analyst expectations in key Asian markets during the first quarter.",
            "publishedAt": (today - timedelta(days=6)).strftime("%Y-%m-%d"),
            "url": "https://example.com/news1"
        },
        {
            "title": "Apple Faces Supply Chain Constraints for MacBook Pro",
            "description": "Production delays at manufacturing plants in Asia are causing supply chain issues for the new MacBook Pro models.",
            "publishedAt": (today - timedelta(days=5)).strftime("%Y-%m-%d"),
            "url": "https://example.com/news2"
        },
        {
            "title": "Apple Announces New AI Features for iOS",
            "description": "At its developer conference, Apple unveiled new AI capabilities that will be integrated into the next iOS update.",
            "publishedAt": (today - timedelta(days=3)).strftime("%Y-%m-%d"),
            "url": "https://example.com/news3"
        },
        {
            "title": "Apple's Services Revenue Hits All-Time High",
            "description": "Apple's services segment, including App Store and Apple Music, reported record revenue growth for the quarter.",
            "publishedAt": (today - timedelta(days=2)).strftime("%Y-%m-%d"),
            "url": "https://example.com/news4"
        },
        {
            "title": "Analysts Neutral on Apple Stock Ahead of Earnings",
            "description": "Wall Street analysts maintain a neutral outlook on Apple stock as the company prepares to release its quarterly earnings report next week.",
            "publishedAt": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            "url": "https://example.com/news5"
        }
    ]
    
    # Analyze the simulated news articles
    print(f"\nAnalyzing simulated news for {ticker}:")
    print("-" * 80)
    
    all_sentiments = []
    
    for article in news_data:
        # Combine title and description for analysis
        text = f"{article['title']} {article['description']}"
        result = sentiment_analyzer.analyze_text(text)
        
        print(f"\nDate: {article['publishedAt']}")
        print(f"Title: {article['title']}")
        print(f"Sentiment: {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        all_sentiments.append(result)
    
    # Calculate the overall sentiment
    if all_sentiments:
        # Weight sentiments by confidence
        sentiment_scores = {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0
        }
        
        for sentiment in all_sentiments:
            sentiment_type = sentiment['sentiment']
            confidence = sentiment['confidence']
            sentiment_scores[sentiment_type] += confidence
        
        # Normalize scores
        total = sum(sentiment_scores.values())
        if total > 0:
            sentiment_scores = {k: v/total for k, v in sentiment_scores.items()}
        
        # Determine overall sentiment
        max_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
        
        print("\n" + "=" * 80)
        print(f"OVERALL SENTIMENT FOR {ticker}:")
        print("-" * 80)
        print(f"Sentiment: {max_sentiment[0].upper()}")
        print(f"Confidence: {max_sentiment[1]:.4f}")
        print(f"Sentiment Distribution: Positive={sentiment_scores['positive']:.4f}, " +
              f"Negative={sentiment_scores['negative']:.4f}, " +
              f"Neutral={sentiment_scores['neutral']:.4f}")
    
    # Create a bar chart of sentiment scores
    if all_sentiments:
        plt.figure(figsize=(10, 6))
        
        # Extract sentiment scores
        dates = [article['publishedAt'] for article in news_data]
        sentiments = [s['sentiment'] for s in all_sentiments]
        confidences = [s['confidence'] for s in all_sentiments]
        
        # Create color mapping
        colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
        bar_colors = [colors[s] for s in sentiments]
        
        # Create bar chart
        plt.bar(range(len(dates)), confidences, color=bar_colors)
        plt.xticks(range(len(dates)), dates, rotation=45)
        plt.xlabel('Date')
        plt.ylabel('Confidence Score')
        plt.title(f'Sentiment Analysis for {ticker} News')
        
        # Add a legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[s], label=s.capitalize()) for s in colors]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        # Save the chart
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/sentiment_analysis_demo.png')
        print(f"\nSentiment chart saved to results/sentiment_analysis_demo.png")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main() 