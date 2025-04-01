import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Union, Any
import logging
import os
import json
import requests
from datetime import datetime, timedelta

class SentimentModel:
    """
    Financial sentiment analysis model using FinBERT
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", cache_dir: str = "models/sentiment_models/"):
        """
        Initialize sentiment analysis model
        
        Args:
            model_name: Name of the pretrained model to use
            cache_dir: Directory to cache model files
        """
        self.logger = logging.getLogger(__name__)
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
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """
        Load pretrained FinBERT model
        """
        try:
            self.logger.info(f"Loading sentiment model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            self.model.to(self.device)
            self.logger.info("Sentiment model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading sentiment model: {e}")
            raise
    
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of a single text
        
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
    
    def analyze_market_sentiment(self, data: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """
        Analyze market sentiment based on price movements and technical indicators
        
        Args:
            data: Historical price data with columns: Date, Close, Volume, etc.
            
        Returns:
            Market sentiment analysis results
        """
        try:
            if len(data) < 2:
                return {
                    "market_sentiment": "neutral",
                    "confidence": 0.0,
                    "sentiment_score": 0.0
                }
            
            # Sort data by date
            data = data.sort_values('Date')
            
            # Calculate basic technical indicators
            # 1. Price movement (up/down)
            price_change = data['Close'].pct_change()
            latest_change = price_change.iloc[-1]
            
            # 2. Volume change
            if 'Volume' in data.columns:
                volume_change = data['Volume'].pct_change()
                latest_volume_change = volume_change.iloc[-1]
                # Volume increase with price increase is bullish
                volume_signal = 1 if (latest_volume_change > 0 and latest_change > 0) or \
                                 (latest_volume_change < 0 and latest_change < 0) else -1
            else:
                volume_signal = 0
            
            # 3. Moving average indicators
            if len(data) >= 20:
                ma_20 = data['Close'].rolling(window=20).mean()
                ma_signal = 1 if data['Close'].iloc[-1] > ma_20.iloc[-1] else -1
            else:
                ma_signal = 0
            
            # 4. RSI indicator if available
            rsi_signal = 0
            for col in data.columns:
                if 'RSI' in col:
                    rsi = data[col].iloc[-1]
                    if rsi < 30:  # Oversold
                        rsi_signal = 1
                    elif rsi > 70:  # Overbought
                        rsi_signal = -1
                    break
            
            # 5. MACD indicator if available
            macd_signal = 0
            macd_col = None
            macd_signal_col = None
            
            for col in data.columns:
                if 'MACD' in col and 'Signal' not in col:
                    macd_col = col
                if 'MACD_Signal' in col:
                    macd_signal_col = col
            
            if macd_col and macd_signal_col:
                macd = data[macd_col].iloc[-1]
                macd_signal_val = data[macd_signal_col].iloc[-1]
                macd_signal = 1 if macd > macd_signal_val else -1
            
            # Combine signals with weights
            weights = {
                'price': 0.4,
                'volume': 0.2,
                'ma': 0.2,
                'rsi': 0.1,
                'macd': 0.1
            }
            
            # Calculate weighted sentiment score (-1 to 1)
            price_sentiment = 1 if latest_change > 0 else -1
            sentiment_score = (
                weights['price'] * price_sentiment +
                weights['volume'] * volume_signal +
                weights['ma'] * ma_signal +
                weights['rsi'] * rsi_signal +
                weights['macd'] * macd_signal
            )
            
            # Convert to sentiment label
            if sentiment_score > 0.3:
                market_sentiment = "positive"
            elif sentiment_score < -0.3:
                market_sentiment = "negative"
            else:
                market_sentiment = "neutral"
            
            # Calculate confidence based on the absolute value of the sentiment score
            confidence = min(abs(sentiment_score), 1.0)
            
            return {
                "market_sentiment": market_sentiment,
                "confidence": confidence,
                "sentiment_score": sentiment_score,
                "indicators": {
                    "price_change": latest_change,
                    "volume_signal": volume_signal,
                    "ma_signal": ma_signal,
                    "rsi_signal": rsi_signal,
                    "macd_signal": macd_signal
                }
            }
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {e}")
            return {
                "market_sentiment": "neutral",
                "confidence": 0.0,
                "sentiment_score": 0.0,
                "indicators": {}
            }
    
    def get_combined_sentiment(self, ticker: str, price_data: pd.DataFrame, days: int = 7) -> Dict[str, Any]:
        """
        Get combined sentiment analysis from news and market data
        
        Args:
            ticker: Stock ticker symbol
            price_data: Historical price data
            days: Number of days of news to analyze
            
        Returns:
            Combined sentiment analysis
        """
        # Get news sentiment
        news_sentiment = self.analyze_news(ticker, days)
        
        # Get market sentiment
        market_sentiment = self.analyze_market_sentiment(price_data)
        
        # Combine sentiments (simple average for demonstration)
        # Map sentiment labels to scores
        sentiment_scores = {
            "positive": 1.0,
            "neutral": 0.0,
            "negative": -1.0
        }
        
        news_score = sentiment_scores.get(news_sentiment["overall_sentiment"], 0.0) * news_sentiment["confidence"]
        market_score = market_sentiment["sentiment_score"]
        
        # Weight market sentiment more heavily than news (60/40)
        combined_score = 0.4 * news_score + 0.6 * market_score
        
        # Convert back to sentiment label
        if combined_score > 0.3:
            combined_sentiment = "positive"
        elif combined_score < -0.3:
            combined_sentiment = "negative"
        else:
            combined_sentiment = "neutral"
        
        # Calculate confidence
        combined_confidence = (0.4 * news_sentiment["confidence"] + 0.6 * market_sentiment["confidence"])
        
        return {
            "ticker": ticker,
            "combined_sentiment": combined_sentiment,
            "confidence": combined_confidence,
            "news_sentiment": news_sentiment,
            "market_sentiment": market_sentiment,
            "sentiment_score": combined_score
        } 