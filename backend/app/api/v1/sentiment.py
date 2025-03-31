"""
Sentiment Analysis API endpoints.

This module provides API endpoints for news sentiment analysis and market sentiment.
"""

from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.model_manager import get_model_manager
from app.utils.database_utils import get_db, cache_query
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{symbol}")
@cache_query(ttl_seconds=1800)  # Cache results for 30 minutes
async def get_sentiment(
    symbol: str,
    days: int = Query(7, description="Number of days of news to analyze"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get sentiment analysis for news related to a stock symbol.
    """
    try:
        logger.info(f"Fetching sentiment for {symbol}, days={days}")
        
        # This would typically:
        # 1. Fetch news articles for the symbol
        # 2. Analyze sentiment using a pre-trained model
        # 3. Aggregate results
        
        # Get the model manager
        model_manager = get_model_manager()
        
        # Placeholder for fetching news
        # In a real implementation, this would fetch news from a news API
        news_articles = await fetch_news_for_symbol(symbol, days)
        
        # Analyze sentiment using the sentiment model
        sentiment_results = []
        if news_articles:
            for article in news_articles:
                # Use the model manager to get predictions
                sentiment = model_manager.predict(
                    model_name="finbert",
                    model_type="sentiment",
                    data=article["title"] + " " + article["summary"]
                )
                
                sentiment_results.append({
                    "title": article["title"],
                    "date": article["date"],
                    "source": article["source"],
                    "sentiment": sentiment["label"],
                    "score": sentiment["score"]
                })
        
        # Calculate aggregate sentiment
        if sentiment_results:
            positive = sum(1 for item in sentiment_results if item["sentiment"] == "positive")
            negative = sum(1 for item in sentiment_results if item["sentiment"] == "negative")
            neutral = sum(1 for item in sentiment_results if item["sentiment"] == "neutral")
            
            total = len(sentiment_results)
            aggregate_score = sum(item["score"] * (1 if item["sentiment"] == "positive" else -1 if item["sentiment"] == "negative" else 0) 
                                for item in sentiment_results) / total
        else:
            positive = negative = neutral = 0
            total = 0
            aggregate_score = 0
        
        return {
            "symbol": symbol,
            "days_analyzed": days,
            "articles_analyzed": total,
            "sentiment_distribution": {
                "positive": positive,
                "neutral": neutral,
                "negative": negative
            },
            "aggregate_score": aggregate_score,
            "sentiment_results": sentiment_results,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing sentiment: {str(e)}"
        )

@router.get("/market/overview")
@cache_query(ttl_seconds=3600)  # Cache results for 1 hour
async def get_market_sentiment(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get overall market sentiment analysis.
    """
    try:
        logger.info("Fetching market sentiment overview")
        
        # This would typically:
        # 1. Fetch news articles about the overall market
        # 2. Analyze sentiment
        # 3. Return aggregated results
        
        # Placeholder - in a real implementation, this would analyze market news
        import random
        
        return {
            "market_sentiment": random.choice(["bullish", "neutral", "bearish"]),
            "confidence": random.uniform(0.6, 0.95),
            "sectors": {
                "technology": random.uniform(-1, 1),
                "healthcare": random.uniform(-1, 1),
                "financials": random.uniform(-1, 1),
                "consumer": random.uniform(-1, 1),
                "energy": random.uniform(-1, 1)
            },
            "articles_analyzed": random.randint(50, 200),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing market sentiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing market sentiment: {str(e)}"
        )

# Helper function to fetch news for a symbol
async def fetch_news_for_symbol(symbol: str, days: int) -> List[Dict[str, Any]]:
    """
    Fetch news articles for a stock symbol.
    
    This is a placeholder implementation that returns mock data.
    In a real implementation, this would fetch news from a news API.
    """
    # Mock data for demonstration
    import random
    from datetime import datetime, timedelta
    
    sources = ["Bloomberg", "Reuters", "CNBC", "The Wall Street Journal", "Financial Times"]
    sentiment_words = {
        "positive": ["surge", "gain", "rise", "profit", "growth", "soar", "up", "strong"],
        "negative": ["fall", "drop", "plunge", "decline", "loss", "weak", "down", "struggle"],
        "neutral": ["announces", "reports", "plans", "states", "updates", "maintains", "holds"]
    }
    
    articles = []
    for i in range(random.randint(5, 15)):
        # Generate a random date within the specified days
        date = datetime.now() - timedelta(days=random.randint(0, days-1), 
                                          hours=random.randint(0, 23),
                                          minutes=random.randint(0, 59))
        
        # Generate a random sentiment
        sentiment = random.choice(["positive", "negative", "neutral"])
        sentiment_word = random.choice(sentiment_words[sentiment])
        
        # Generate a random title
        if sentiment == "positive":
            title = f"{symbol} shares {sentiment_word} on strong earnings report"
        elif sentiment == "negative":
            title = f"{symbol} stock {sentiment_word} after quarterly results"
        else:
            title = f"{symbol} {sentiment_word} new strategic initiative"
        
        articles.append({
            "title": title,
            "summary": f"This is a mock summary for {symbol} news article.",
            "date": date.isoformat(),
            "source": random.choice(sources),
            "url": f"https://example.com/news/{symbol.lower()}/{i}"
        })
    
    # Sort by date, newest first
    articles.sort(key=lambda x: x["date"], reverse=True)
    
    return articles 