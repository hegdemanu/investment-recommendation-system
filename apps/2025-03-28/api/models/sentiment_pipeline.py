#!/usr/bin/env python3
"""
Sentiment Analysis Pipeline for Investment Recommendation System
Implements FinBERT-based sentiment analysis for financial texts
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """Initialize the sentiment analyzer with FinBERT model"""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading FinBERT model from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
        self.labels = ["negative", "neutral", "positive"]
        
    async def analyze_text(self, text: str) -> Dict[str, Union[str, float]]:
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
    
    async def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[Dict[str, Union[str, float]]]:
        """Analyze sentiment for a batch of texts"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Process each text in batch
            for text, probs in zip(batch_texts, probabilities):
                prediction = torch.argmax(probs).item()
                confidence = torch.max(probs).item()
                
                results.append({
                    "text": text,
                    "sentiment": self.labels[prediction],
                    "confidence": confidence,
                    "probabilities": {
                        label: prob.item()
                        for label, prob in zip(self.labels, probs)
                    }
                })
        
        return results
    
    def calculate_sentiment_score(
        self,
        sentiment_results: List[Dict[str, Union[str, float]]],
        weights: Dict[str, float] = None
    ) -> float:
        """Calculate weighted sentiment score from analysis results"""
        if weights is None:
            weights = {
                "negative": -1.0,
                "neutral": 0.0,
                "positive": 1.0
            }
        
        total_score = 0.0
        total_confidence = 0.0
        
        for result in sentiment_results:
            sentiment = result["sentiment"]
            confidence = result["confidence"]
            
            total_score += weights[sentiment] * confidence
            total_confidence += confidence
        
        return total_score / total_confidence if total_confidence > 0 else 0.0
    
    def adjust_prediction_weights(
        self,
        sentiment_score: float,
        base_weights: Dict[str, float],
        sentiment_impact: float = 0.2
    ) -> Dict[str, float]:
        """Adjust model weights based on sentiment score"""
        # Normalize sentiment score to [-1, 1]
        normalized_score = max(min(sentiment_score, 1.0), -1.0)
        
        # Calculate adjustment factor
        adjustment = normalized_score * sentiment_impact
        
        # Adjust weights
        adjusted_weights = {}
        for model, weight in base_weights.items():
            if model == "lstm":
                # Increase LSTM weight for positive sentiment
                adjusted_weights[model] = weight * (1 + adjustment)
            elif model == "arima_garch":
                # Increase ARIMA+GARCH weight for negative sentiment
                adjusted_weights[model] = weight * (1 - adjustment)
            else:
                # Keep Prophet weight relatively stable
                adjusted_weights[model] = weight
        
        # Normalize weights to sum to 1
        total_weight = sum(adjusted_weights.values())
        return {
            model: weight / total_weight
            for model, weight in adjusted_weights.items()
        }
    
    async def analyze_news_impact(
        self,
        news_items: List[Dict[str, str]],
        current_price: float
    ) -> Dict[str, Union[float, List[Dict]]]:
        """Analyze news impact on price predictions"""
        # Analyze sentiment for all news items
        sentiment_results = await self.analyze_batch([item["text"] for item in news_items])
        
        # Calculate overall sentiment score
        sentiment_score = self.calculate_sentiment_score(sentiment_results)
        
        # Estimate price impact
        price_impact = current_price * (sentiment_score * 0.02)  # 2% max impact
        
        return {
            "sentiment_score": sentiment_score,
            "price_impact": price_impact,
            "detailed_analysis": [
                {
                    **result,
                    "date": news_items[i]["date"],
                    "source": news_items[i]["source"]
                }
                for i, result in enumerate(sentiment_results)
            ]
        }
    
    def save_sentiment_analysis(
        self,
        analysis_results: Dict[str, Union[float, List[Dict]]],
        save_dir: str = "results/sentiment/"
    ):
        """Save sentiment analysis results"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = save_path / f"sentiment_analysis_{timestamp}.json"
        
        # Save results
        with open(filename, "w") as f:
            pd.json_normalize(analysis_results).to_json(f, indent=2)
            
        logger.info(f"Sentiment analysis results saved to {filename}") 