"""
API Routes for Investment Recommendation System
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import our pipelines
from app.models.training_pipeline import ModelTrainingPipeline
from app.models.prediction_pipeline import PredictionPipeline
from app.models.sentiment_pipeline import SentimentAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    symbol: str
    horizon: int = Field(default=30, ge=1, le=365)
    include_sentiment: bool = Field(default=True)

class NewsItem(BaseModel):
    text: str
    date: str
    source: str

class SentimentRequest(BaseModel):
    texts: List[str]
    weights: Optional[Dict[str, float]] = None

class TrainingRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    target_col: str = "Close"

# Dependencies
async def get_sentiment_analyzer():
    return SentimentAnalyzer()

async def get_prediction_pipeline():
    return PredictionPipeline()

async def get_training_pipeline():
    return ModelTrainingPipeline()

# API endpoints
@router.post("/predict")
async def predict(
    request: PredictionRequest,
    prediction_pipeline: PredictionPipeline = Depends(get_prediction_pipeline),
    sentiment_analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    """Generate price predictions for a given symbol"""
    try:
        # Load historical data (implement data loading logic)
        data = pd.read_csv(f"data/{request.symbol}.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # Generate predictions
        predictions = prediction_pipeline.generate_ensemble_predictions(
            data=data,
            horizon=request.horizon
        )
        
        if request.include_sentiment:
            # Get recent news (implement news fetching logic)
            news_items = [
                {"text": "Sample news", "date": "2024-03-26", "source": "Example"}
            ]
            
            # Analyze sentiment impact
            sentiment_impact = await sentiment_analyzer.analyze_news_impact(
                news_items=news_items,
                current_price=data['Close'].iloc[-1]
            )
            
            # Adjust predictions based on sentiment
            adjusted_weights = sentiment_analyzer.adjust_prediction_weights(
                sentiment_score=sentiment_impact['sentiment_score'],
                base_weights=predictions['ensemble']['weights']
            )
            
            predictions['sentiment_adjusted'] = {
                'weights': adjusted_weights,
                'impact': sentiment_impact
            }
        
        return predictions
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sentiment/analyze")
async def analyze_sentiment(
    request: SentimentRequest,
    sentiment_analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    """Analyze sentiment for given texts"""
    try:
        results = await sentiment_analyzer.analyze_batch(request.texts)
        
        if request.weights:
            score = sentiment_analyzer.calculate_sentiment_score(
                results,
                weights=request.weights
            )
            return {"detailed_results": results, "overall_score": score}
        
        return {"detailed_results": results}
    
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/train")
async def train_models(
    request: TrainingRequest,
    training_pipeline: ModelTrainingPipeline = Depends(get_training_pipeline)
):
    """Train prediction models for a given symbol"""
    try:
        # Load training data
        data = pd.read_csv(f"data/{request.symbol}.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # Filter data by date range
        mask = (
            (data.index >= request.start_date) &
            (data.index <= request.end_date)
        )
        training_data = data.loc[mask]
        
        # Train models
        results = training_pipeline.train_all_models(
            data=training_data,
            target_col=request.target_col
        )
        
        # Save models
        training_pipeline.save_models(
            save_dir=f"models/{request.symbol}/"
        )
        
        return {
            "status": "success",
            "symbol": request.symbol,
            "training_results": results
        }
    
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    } 