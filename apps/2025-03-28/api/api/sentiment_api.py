"""
API endpoints for sentiment analysis capabilities.

This module provides API endpoints for:
- Sentiment analysis on financial text
- Sentiment-weighted predictions
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional

# Import our sentiment analyzer
from app.models.sentiment_pipeline import SentimentAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class BatchSentimentRequest(BaseModel):
    texts: List[str]
    weights: Optional[Dict[str, float]] = None

class BatchSentimentResponse(BaseModel):
    detailed_results: List[Dict]
    overall_score: Optional[float] = None

# Dependencies
async def get_sentiment_analyzer():
    return SentimentAnalyzer()

# API endpoints
@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    sentiment_analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    """Analyze sentiment of a single text"""
    try:
        result = await sentiment_analyzer.analyze_text(request.text)
        return result
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchSentimentResponse)
async def analyze_batch_sentiment(
    request: BatchSentimentRequest,
    sentiment_analyzer: SentimentAnalyzer = Depends(get_sentiment_analyzer)
):
    """Analyze sentiment for multiple texts"""
    try:
        results = await sentiment_analyzer.analyze_batch(request.texts)
        
        response = {"detailed_results": results}
        
        if request.weights:
            score = sentiment_analyzer.calculate_sentiment_score(
                results,
                weights=request.weights
            )
            response["overall_score"] = score
        
        return response
    except Exception as e:
        logger.error(f"Batch sentiment analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 