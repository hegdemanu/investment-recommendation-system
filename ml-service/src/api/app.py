#!/usr/bin/env python3
"""
FastAPI Application for ML Service
Provides API endpoints for model predictions, training, and management
"""

import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import logging
import json
from pydantic import BaseModel, Field

# Local imports
from ..models.prediction_pipeline import PredictionPipeline
from ..models.model_trainer import ModelTrainer
from ..models.sentiment_analyzer import SentimentAnalyzer
from ..utils.model_registry import ModelRegistry
from ..utils.data_processor import DataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Investment Recommendation System ML API",
    description="API for ML model predictions, training, and management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_registry = ModelRegistry()
data_processor = DataProcessor()
prediction_pipeline = PredictionPipeline(model_registry=model_registry)
model_trainer = ModelTrainer(model_registry=model_registry)
sentiment_analyzer = SentimentAnalyzer(model_registry=model_registry)

# ----- Pydantic Models -----

class PredictionRequest(BaseModel):
    symbol: str
    modelType: Optional[str] = None
    horizon: Optional[int] = 5
    includeMetadata: Optional[bool] = False

class TrainingRequest(BaseModel):
    symbol: str
    modelType: str
    options: Optional[Dict[str, Any]] = None

class RAGQueryRequest(BaseModel):
    query: str
    context: Optional[str] = None

# ----- API Routes -----

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "ml-service"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Generate predictions for a symbol"""
    try:
        result = prediction_pipeline.generate_predictions(
            symbol=request.symbol,
            horizon=request.horizon,
            model_type=request.modelType
        )
        
        # Include model metadata if requested
        if request.includeMetadata and result.get("modelType"):
            metadata = None
            model_type = result["modelType"]
            
            # Get latest model metadata
            model_metadata = model_registry.get_models_by_symbol(request.symbol)
            for model in model_metadata:
                if model["modelType"] == model_type:
                    metadata = model
                    break
            
            if metadata:
                result["metadata"] = metadata
        
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Retrain a model for a symbol"""
    try:
        # Start training in background
        background_tasks.add_task(
            model_trainer.train,
            symbol=request.symbol,
            model_type=request.modelType,
            options=request.options
        )
        
        return {
            "status": "training_started",
            "message": f"Model training for {request.symbol} with {request.modelType} started in background",
            "symbol": request.symbol,
            "modelType": request.modelType
        }
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get all available models"""
    try:
        # This is a placeholder - we would need to implement a method to get all models
        all_symbols = set()
        models_data = []
        
        # For now, let's simulate some data
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        for symbol in symbols:
            model_metadata = model_registry.get_models_by_symbol(symbol)
            models_data.extend(model_metadata)
            all_symbols.add(symbol)
        
        return models_data
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/symbol/{symbol}")
async def get_models_by_symbol(symbol: str):
    """Get models for a specific symbol"""
    try:
        models = model_registry.get_models_by_symbol(symbol)
        return models
    except Exception as e:
        logger.error(f"Error getting models for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/compare/{symbol}")
async def compare_models(symbol: str):
    """Compare models for a symbol"""
    try:
        comparison = model_registry.compare_models(symbol)
        return comparison
    except Exception as e:
        logger.error(f"Error comparing models for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a model by ID"""
    try:
        success = model_registry.delete_model(model_id)
        if success:
            return {"status": "success", "message": f"Model {model_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment/{symbol}")
async def analyze_sentiment(symbol: str):
    """Analyze sentiment for a symbol"""
    try:
        result = await sentiment_analyzer.analyze(symbol)
        return result
    except Exception as e:
        logger.error(f"Sentiment analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment/{symbol}/detailed")
async def get_detailed_sentiment(symbol: str):
    """Get detailed sentiment analysis for a symbol"""
    try:
        result = await sentiment_analyzer.get_detailed_sentiment(symbol)
        return result
    except Exception as e:
        logger.error(f"Detailed sentiment analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query")
async def query_rag(request: RAGQueryRequest):
    """Query the RAG system"""
    try:
        # This is a placeholder - we would need to implement a RAG system
        # For now, let's return a simulated response
        return {
            "query": request.query,
            "response": f"This is a simulated response to: {request.query}",
            "sources": [
                {
                    "title": "Financial Times",
                    "url": "https://www.ft.com",
                    "relevance": 0.95,
                    "snippet": "Relevant information from Financial Times..."
                },
                {
                    "title": "Wall Street Journal",
                    "url": "https://www.wsj.com",
                    "relevance": 0.85,
                    "snippet": "Relevant information from Wall Street Journal..."
                }
            ],
            "relatedSymbols": ["AAPL", "MSFT", "GOOGL"] if "tech" in request.query.lower() else ["JPM", "BAC", "GS"]
        }
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/selector/{symbol}")
async def get_model_selector(symbol: str):
    """Get model recommendation for a symbol"""
    try:
        comparison = model_registry.compare_models(symbol)
        if comparison and "recommendedModel" in comparison:
            return {"recommendedModel": comparison["recommendedModel"]}
        return {"recommendedModel": "LSTM"}  # Default fallback
    except Exception as e:
        logger.error(f"Error getting model recommendation for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True) 