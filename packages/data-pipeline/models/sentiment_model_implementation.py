import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Union, Any
import logging
import os
from datetime import datetime, timedelta
import requests
import json

class SentimentAnalyzer:
    """
    Financial sentiment analysis model using pre-trained FinBERT
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", cache_dir: str = "trading_engine/models/sentiment_models/"):
        """
        Initialize sentiment analysis model with pre-trained FinBERT
        
        Args:
            model_name: Name of the pretrained model to use
            cache_dir: Directory to cache model files
        """
        self.logger = self._setup_logger()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Define sentiment labels
        self.sentiment_labels = {
            0: "positive",
            1: "negative",
            2: "neutral"
        }
        
        # Log the initialization
        self.logger.info(f"Initializing sentiment analyzer with model: {model_name} on device: {self.device}")
        
        # Load model
        self.load_model()
    
    def _setup_logger(self):
        """Set up a logger for the sentiment analyzer"""
        logger = logging.getLogger("SentimentAnalyzer")
        logger.setLevel(logging.INFO)
        
        # Check if handler already exists to avoid duplicate logs
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_model(self):
        """
        Load pretrained FinBERT model
        """
        try:
            self.logger.info(f"Loading FinBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model.to(self.device)
            self.logger.info("FinBERT model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading FinBERT model: {e}")
            raise
    
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of a single text using FinBERT
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get predicted label and confidence
            predicted_label = torch.argmax(predictions).item()
            confidence = predictions[0][predicted_label].item()
            
            return {
                "sentiment": self.sentiment_labels[predicted_label],
                "confidence": confidence,
                "raw_scores": {
                    "positive": predictions[0][0].item(),
                    "negative": predictions[0][1].item(),
                    "neutral": predictions[0][2].item()
                }
            }
        except Exception as e:
            self.logger.error(f"Error analyzing text: {e}")
            return {
                "sentiment": "error",
                "confidence": 0.0,
                "raw_scores": {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            }
    
    def analyze_texts(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Analyze sentiment for a list of texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        for text in texts:
            result = self.analyze_text(text)
            results.append(result)
        return results
    
    def analyze_news(self, ticker: str, days: int = 7, api_key: str = None) -> Dict[str, Any]:
        """
        Analyze sentiment from news articles for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of news to analyze
            api_key: News API key (optional)
            
        Returns:
            Aggregated sentiment analysis results
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Use API key from environment if not provided
            if api_key is None:
                api_key = os.environ.get("NEWS_API_KEY")
                if not api_key:
                    self.logger.warning("No NEWS_API_KEY provided or found in environment")
                    return self._empty_news_result(ticker)
            
            # Fetch news articles
            news_url = "https://newsapi.org/v2/everything"
            params = {
                "q": ticker,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "publishedAt",
                "apiKey": api_key
            }
            
            self.logger.info(f"Fetching news for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            response = requests.get(news_url, params=params)
            
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch news: {response.text}")
                return self._empty_news_result(ticker)
            
            news_data = response.json()
            articles = news_data.get("articles", [])
            
            if not articles:
                self.logger.warning(f"No news articles found for {ticker}")
                return self._empty_news_result(ticker)
            
            # Analyze sentiment for each article
            sentiments = []
            for article in articles:
                # Combine title and description for analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self.analyze_text(text)
                sentiments.append({
                    "date": article.get("publishedAt", ""),
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "sentiment": sentiment
                })
            
            # Calculate aggregated sentiment
            if sentiments:
                # Weight sentiments by confidence
                weighted_scores = {
                    "positive": sum(s["sentiment"]["raw_scores"]["positive"] * s["sentiment"]["confidence"] 
                                  for s in sentiments),
                    "negative": sum(s["sentiment"]["raw_scores"]["negative"] * s["sentiment"]["confidence"] 
                                  for s in sentiments),
                    "neutral": sum(s["sentiment"]["raw_scores"]["neutral"] * s["sentiment"]["confidence"] 
                                 for s in sentiments)
                }
                
                # Normalize scores
                total = sum(weighted_scores.values())
                if total > 0:
                    weighted_scores = {k: v/total for k, v in weighted_scores.items()}
                
                # Determine overall sentiment
                max_sentiment = max(weighted_scores.items(), key=lambda x: x[1])
                
                return {
                    "ticker": ticker,
                    "overall_sentiment": max_sentiment[0],
                    "confidence": max_sentiment[1],
                    "article_count": len(sentiments),
                    "sentiment_distribution": weighted_scores,
                    "articles": sentiments
                }
            else:
                return self._empty_news_result(ticker)
        except Exception as e:
            self.logger.error(f"Error analyzing news for {ticker}: {e}")
            return self._empty_news_result(ticker)
    
    def _empty_news_result(self, ticker: str) -> Dict[str, Any]:
        """
        Return empty result structure when news analysis fails
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Empty sentiment analysis results
        """
        return {
            "ticker": ticker,
            "overall_sentiment": "neutral",
            "confidence": 0.0,
            "article_count": 0,
            "sentiment_distribution": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            "articles": []
        }
    
    def save_model_info(self, path: str = None):
        """
        Save model information to JSON file
        
        Args:
            path: Path to save model info
        """
        if path is None:
            path = os.path.join(self.cache_dir, "model_info.json")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "labels": self.sentiment_labels,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(path, 'w') as f:
            json.dump(model_info, f, indent=4)
            
        self.logger.info(f"Model info saved to {path}") 