from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List, Dict, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import requests
import json
import os

class SentimentAnalyzer:
    def __init__(self, model_registry):
        self.model_registry = model_registry
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def _load_model_components(self) -> None:
        """Load the sentiment analysis model and tokenizer"""
        try:
            model_path = self.model_registry.get_model_path("sentiment", "finbert", "model")
            if not model_path:
                # If not in registry, use default FinBERT
                model_path = "ProsusAI/finbert"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            
        except Exception as e:
            raise Exception(f"Error loading sentiment model: {str(e)}")
    
    def _get_news_data(self, symbol: str, days: int = 7) -> List[str]:
        """Fetch recent news articles for a symbol"""
        try:
            # Get company info
            ticker = yf.Ticker(symbol)
            
            # Get news from Yahoo Finance
            news = ticker.news
            
            # Filter recent news
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_news = []
            
            for article in news:
                if datetime.fromtimestamp(article['providerPublishTime']) > cutoff_date:
                    recent_news.append(article['title'] + ". " + article.get('summary', ''))
            
            return recent_news
            
        except Exception as e:
            raise Exception(f"Error fetching news data: {str(e)}")
    
    def _analyze_text(self, texts: List[str]) -> np.ndarray:
        """Analyze sentiment of text using FinBERT"""
        try:
            if not self.model or not self.tokenizer:
                self._load_model_components()
            
            # Tokenize texts
            encoded = self.tokenizer(texts, 
                                   padding=True, 
                                   truncation=True, 
                                   max_length=512,
                                   return_tensors="pt")
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**encoded)
                predictions = torch.softmax(outputs.logits, dim=1)
            
            # Convert to numpy
            predictions = predictions.cpu().numpy()
            
            return predictions
            
        except Exception as e:
            raise Exception(f"Error analyzing text: {str(e)}")
    
    def analyze(self, symbol: str) -> float:
        """
        Analyze sentiment for a symbol
        Returns: sentiment score between -1 (negative) and 1 (positive)
        """
        try:
            # Get recent news
            news_texts = self._get_news_data(symbol)
            
            if not news_texts:
                return 0.0  # Neutral if no news
            
            # Get sentiment predictions
            predictions = self._analyze_text(news_texts)
            
            # Average sentiment scores
            # FinBERT classes: negative (0), neutral (1), positive (2)
            sentiment_scores = []
            for pred in predictions:
                score = pred[2] - pred[0]  # positive - negative
                sentiment_scores.append(score)
            
            # Return average sentiment
            return float(np.mean(sentiment_scores))
            
        except Exception as e:
            raise Exception(f"Sentiment analysis error: {str(e)}")
    
    def get_detailed_sentiment(self, symbol: str) -> Dict:
        """Get detailed sentiment analysis including individual article scores"""
        try:
            news_texts = self._get_news_data(symbol)
            if not news_texts:
                return {"overall": 0.0, "articles": []}
            
            predictions = self._analyze_text(news_texts)
            
            articles = []
            for text, pred in zip(news_texts, predictions):
                score = float(pred[2] - pred[0])
                articles.append({
                    "text": text[:200] + "...",  # Truncate text
                    "sentiment": score,
                    "probabilities": {
                        "negative": float(pred[0]),
                        "neutral": float(pred[1]),
                        "positive": float(pred[2])
                    }
                })
            
            return {
                "overall": float(np.mean([a["sentiment"] for a in articles])),
                "articles": articles
            }
            
        except Exception as e:
            raise Exception(f"Detailed sentiment analysis error: {str(e)}") 