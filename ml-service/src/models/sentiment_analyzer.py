#!/usr/bin/env python3
"""
Sentiment Analysis for Investment Recommendation System
Implements FinBERT-based sentiment analysis for financial texts
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Union, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import requests
import json
import asyncio
from pathlib import Path

# Local imports
from ..utils.model_registry import ModelRegistry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_registry: ModelRegistry = None, model_name: str = "ProsusAI/finbert"):
        """Initialize the sentiment analyzer with model registry"""
        self.model_registry = model_registry or ModelRegistry()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_model_components()
        
    def _load_model_components(self):
        """Load model components from registry or download them"""
        # Check if we have a sentiment model in the registry
        sentiment_model = self.model_registry.get_latest_model(
            "MARKET", "SENTIMENT"
        )
        
        if sentiment_model:
            logger.info("Loading sentiment model from registry")
            self.model = sentiment_model['model']
            self.tokenizer = sentiment_model['tokenizer']
        else:
            logger.info(f"Loading FinBERT model from {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
        self.model.to(self.device)
        self.labels = ["negative", "neutral", "positive"]
    
    async def _get_news_data(self, symbol: str, days: int = 7) -> List[Dict]:
        """Fetch recent news for a stock symbol"""
        # This would typically connect to a news API
        # For demonstration, we'll create some sample data
        try:
            # Try to get news from an external API - replace with your API
            api_key = "your_news_api_key"  # Should be stored securely
            url = f"https://newsapi.org/v2/everything?q={symbol}+stock&from={days}d&language=en&apiKey={api_key}"
            
            # Placeholder - actual implementation would use the API
            return [
                {
                    "title": f"{symbol} announces strong quarterly results",
                    "text": f"{symbol} has reported earnings that exceeded analyst expectations, with revenue growing by 12% year over year.",
                    "date": (datetime.now() - pd.Timedelta(days=1)).isoformat(),
                    "source": "Financial News",
                    "url": f"https://example.com/news/{symbol}/earnings"
                },
                {
                    "title": f"Analysts upgrade {symbol} stock rating",
                    "text": f"Several analysts have upgraded {symbol} to a 'buy' rating, citing strong growth prospects and competitive positioning.",
                    "date": (datetime.now() - pd.Timedelta(days=2)).isoformat(),
                    "source": "Market Watch",
                    "url": f"https://example.com/news/{symbol}/upgrade"
                },
                {
                    "title": f"{symbol} faces regulatory scrutiny",
                    "text": f"Regulatory authorities have announced an investigation into {symbol}'s business practices, creating uncertainty for investors.",
                    "date": (datetime.now() - pd.Timedelta(days=3)).isoformat(),
                    "source": "Business Insights",
                    "url": f"https://example.com/news/{symbol}/regulation"
                }
            ]
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    async def _analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """Analyze sentiment of a single text"""
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get prediction and confidence
        prediction = torch.argmax(probabilities).item()
        confidence = torch.max(probabilities).item()
        
        return {
            "text": text,
            "sentiment": self.labels[prediction],
            "confidence": confidence,
            "probabilities": {
                label: prob.item()
                for label, prob in zip(self.labels, probabilities[0])
            }
        }
    
    async def analyze(self, symbol: str) -> Dict[str, Union[str, float, List]]:
        """Analyze sentiment for a stock symbol"""
        # Get news data
        news_items = await self._get_news_data(symbol)
        
        if not news_items:
            return {
                "symbol": symbol,
                "overallSentiment": 0,
                "sentimentLabel": "NEUTRAL",
                "confidence": 0.5,
                "newsItems": []
            }
        
        # Analyze each news item
        analysis_tasks = [
            self._analyze_text(item["text"]) for item in news_items
        ]
        sentiment_results = await asyncio.gather(*analysis_tasks)
        
        # Map sentiment labels to numerical values
        sentiment_values = {
            "negative": -1.0,
            "neutral": 0.0,
            "positive": 1.0
        }
        
        # Calculate average sentiment score
        total_confidence = 0.0
        weighted_sentiment = 0.0
        
        for result in sentiment_results:
            sentiment = result["sentiment"]
            confidence = result["confidence"]
            
            weighted_sentiment += sentiment_values[sentiment] * confidence
            total_confidence += confidence
        
        if total_confidence > 0:
            avg_sentiment = weighted_sentiment / total_confidence
        else:
            avg_sentiment = 0.0
        
        # Determine sentiment label
        if avg_sentiment > 0.2:
            sentiment_label = "BULLISH"
        elif avg_sentiment < -0.2:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"
        
        # Prepare newsItems for response
        news_with_sentiment = []
        for i, result in enumerate(sentiment_results):
            news_with_sentiment.append({
                "title": news_items[i]["title"],
                "url": news_items[i].get("url", ""),
                "source": news_items[i]["source"],
                "date": news_items[i]["date"],
                "sentiment": sentiment_values[result["sentiment"]],
                "relevance": result["confidence"]
            })
        
        return {
            "symbol": symbol,
            "overallSentiment": float(avg_sentiment),
            "sentimentLabel": sentiment_label,
            "confidence": float(total_confidence / len(sentiment_results)) if sentiment_results else 0.5,
            "newsItems": news_with_sentiment
        }
    
    async def get_detailed_sentiment(self, symbol: str) -> Dict[str, any]:
        """Get detailed sentiment analysis including individual article scores and probabilities"""
        # Get news data
        news_items = await self._get_news_data(symbol)
        
        if not news_items:
            return {
                "symbol": symbol,
                "overallSentiment": 0,
                "sentimentLabel": "NEUTRAL",
                "confidence": 0.5,
                "newsItems": [],
                "sentimentBreakdown": {
                    "negative": 0,
                    "neutral": 1.0,
                    "positive": 0
                }
            }
        
        # Analyze each news item
        analysis_tasks = [
            self._analyze_text(item["text"]) for item in news_items
        ]
        sentiment_results = await asyncio.gather(*analysis_tasks)
        
        # Calculate aggregate sentiment metrics
        sentiment_counts = {"negative": 0, "neutral": 0, "positive": 0}
        total_confidence = 0.0
        
        for result in sentiment_results:
            sentiment = result["sentiment"]
            confidence = result["confidence"]
            
            sentiment_counts[sentiment] += 1
            total_confidence += confidence
        
        # Calculate probability distribution
        sentiment_distribution = {
            sentiment: count / len(sentiment_results)
            for sentiment, count in sentiment_counts.items()
        }
        
        # Calculate weighted sentiment score
        sentiment_values = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
        weighted_sentiment = sum(
            sentiment_values[result["sentiment"]] * result["confidence"]
            for result in sentiment_results
        ) / total_confidence if total_confidence > 0 else 0.0
        
        # Determine sentiment label
        if weighted_sentiment > 0.2:
            sentiment_label = "BULLISH"
        elif weighted_sentiment < -0.2:
            sentiment_label = "BEARISH"
        else:
            sentiment_label = "NEUTRAL"
        
        # Prepare detailed news items
        detailed_news = []
        for i, result in enumerate(sentiment_results):
            detailed_news.append({
                "title": news_items[i]["title"],
                "url": news_items[i].get("url", ""),
                "source": news_items[i]["source"],
                "date": news_items[i]["date"],
                "sentiment": sentiment_values[result["sentiment"]],
                "sentimentLabel": result["sentiment"].upper(),
                "confidence": result["confidence"],
                "probabilities": result["probabilities"]
            })
        
        return {
            "symbol": symbol,
            "overallSentiment": float(weighted_sentiment),
            "sentimentLabel": sentiment_label,
            "confidence": float(total_confidence / len(sentiment_results)),
            "sentimentBreakdown": sentiment_distribution,
            "newsItems": detailed_news
        } 