from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Union
import logging

class SentimentAnalyzer:
    """
    Module for analyzing financial sentiment using FinBERT.
    Provides both news-based and market-based sentiment analysis.
    """
    
    def __init__(self):
        """Initialize the SentimentAnalyzer with FinBERT model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "ProsusAI/finbert"
        
        print("Loading FinBERT model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        print("FinBERT model loaded successfully.")
        
        # Define sentiment labels
        self.sentiment_labels = {
            0: "positive",
            1: "negative",
            2: "neutral"
        }
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of a single text using FinBERT.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict : Sentiment analysis results
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
            self.logger.error(f"Error analyzing text: {str(e)}")
            return {
                "sentiment": "error",
                "confidence": 0.0,
                "raw_scores": {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            }
    
    def analyze_news(self, ticker: str, days: int = 7) -> Dict[str, Union[str, float, List[Dict]]]:
        """
        Analyze sentiment from news articles for a specific ticker.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        days : int, optional
            Number of days of news to analyze
            
        Returns:
        --------
        dict : Aggregated sentiment analysis results
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch news articles (example using NewsAPI)
            # Note: You'll need to replace this with your preferred news API
            news_url = f"https://newsapi.org/v2/everything"
            params = {
                "q": ticker,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "language": "en",
                "sortBy": "publishedAt",
                "apiKey": "YOUR_NEWS_API_KEY"  # Replace with your API key
            }
            
            response = requests.get(news_url, params=params)
            if response.status_code != 200:
                raise Exception(f"Failed to fetch news: {response.text}")
            
            news_data = response.json()
            articles = news_data.get("articles", [])
            
            # Analyze sentiment for each article
            sentiments = []
            for article in articles:
                # Combine title and description for analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self.analyze_text(text)
                sentiments.append({
                    "date": article.get("publishedAt", ""),
                    "title": article.get("title", ""),
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
                return {
                    "ticker": ticker,
                    "overall_sentiment": "neutral",
                    "confidence": 0.0,
                    "article_count": 0,
                    "sentiment_distribution": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                    "articles": []
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing news for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "overall_sentiment": "error",
                "confidence": 0.0,
                "article_count": 0,
                "sentiment_distribution": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
                "articles": []
            }
    
    def analyze_market_sentiment(self, data: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """
        Analyze market sentiment based on price movements and technical indicators.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical price data with columns: Date, Price, Volume, etc.
            
        Returns:
        --------
        dict : Market sentiment analysis results
        """
        try:
            # Calculate basic technical indicators
            data = data.sort_values('Date')
            
            # Calculate returns
            data['Returns'] = data['Price'].pct_change()
            
            # Calculate volatility (20-day rolling)
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            
            # Calculate momentum (10-day)
            data['Momentum'] = data['Price'].pct_change(10)
            
            # Calculate volume trend
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Trend'] = (data['Volume'] / data['Volume_MA'] - 1)
            
            # Get latest values
            latest = data.iloc[-1]
            
            # Define sentiment rules
            sentiment_score = 0
            
            # Price momentum
            if latest['Momentum'] > 0.02:  # 2% positive momentum
                sentiment_score += 1
            elif latest['Momentum'] < -0.02:  # 2% negative momentum
                sentiment_score -= 1
            
            # Volume trend
            if latest['Volume_Trend'] > 0.2:  # 20% above average volume
                sentiment_score += 1
            elif latest['Volume_Trend'] < -0.2:  # 20% below average volume
                sentiment_score -= 1
            
            # Volatility
            if latest['Volatility'] > data['Volatility'].mean():
                sentiment_score -= 0.5  # High volatility suggests uncertainty
            
            # Determine overall sentiment
            if sentiment_score > 1:
                sentiment = "positive"
            elif sentiment_score < -1:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Calculate confidence based on strength of signals
            confidence = min(abs(sentiment_score) / 3, 1.0)
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "score": sentiment_score,
                "metrics": {
                    "momentum": latest['Momentum'],
                    "volume_trend": latest['Volume_Trend'],
                    "volatility": latest['Volatility']
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {str(e)}")
            return {
                "sentiment": "error",
                "confidence": 0.0,
                "score": 0.0,
                "metrics": {}
            }
    
    def get_combined_sentiment(self, ticker: str, price_data: pd.DataFrame, days: int = 7) -> Dict:
        """
        Combine news and market sentiment analysis.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        price_data : pd.DataFrame
            Historical price data
        days : int, optional
            Number of days of news to analyze
            
        Returns:
        --------
        dict : Combined sentiment analysis results
        """
        try:
            # Get news sentiment
            news_sentiment = self.analyze_news(ticker, days)
            
            # Get market sentiment
            market_sentiment = self.analyze_market_sentiment(price_data)
            
            # Combine sentiments with weights
            news_weight = 0.6  # Weight for news sentiment
            market_weight = 0.4  # Weight for market sentiment
            
            # Convert sentiments to numerical scores
            sentiment_scores = {
                "positive": 1,
                "negative": -1,
                "neutral": 0,
                "error": 0
            }
            
            news_score = sentiment_scores[news_sentiment["overall_sentiment"]] * news_sentiment["confidence"]
            market_score = sentiment_scores[market_sentiment["sentiment"]] * market_sentiment["confidence"]
            
            # Calculate combined score
            combined_score = news_score + market_score
            
            # Determine overall sentiment
            if combined_score > 0.3:
                overall_sentiment = "positive"
            elif combined_score < -0.3:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            # Calculate overall confidence
            overall_confidence = min(abs(combined_score), 1.0)
            
            return {
                "ticker": ticker,
                "overall_sentiment": overall_sentiment,
                "overall_confidence": overall_confidence,
                "combined_score": combined_score,
                "news_sentiment": news_sentiment,
                "market_sentiment": market_sentiment
            }
            
        except Exception as e:
            self.logger.error(f"Error getting combined sentiment: {str(e)}")
            return {
                "ticker": ticker,
                "overall_sentiment": "error",
                "overall_confidence": 0.0,
                "combined_score": 0.0,
                "news_sentiment": {},
                "market_sentiment": {}
            } 