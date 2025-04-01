#!/usr/bin/env python3
"""
FastAPI Application for ML Service
Provides API endpoints for model predictions, training, and management
"""

import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any, Union
import logging
import json
from pydantic import BaseModel, Field
from enum import Enum
import datetime

# Local imports
from ..models.prediction_pipeline import PredictionPipeline
from ..models.model_trainer import ModelTrainer
from ..models.sentiment_analyzer import SentimentAnalyzer
from ..utils.model_registry import ModelRegistry
from ..utils.data_processor import DataProcessor
from ..models.auto_model_selector import AutoModelSelector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Investment Recommendation System ML API",
    description="API for ML model predictions, training, and management",
    version="2.0.0"
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
auto_model_selector = AutoModelSelector(model_registry=model_registry)

# ----- Pydantic Models and Enums -----

class ModelType(str, Enum):
    LSTM = "lstm"
    ARIMA = "arima"
    PROPHET = "prophet"
    AUTO = "auto"  # Let the system decide the best model

class TimeFrame(str, Enum):
    DAY_1 = "1d"
    WEEK_1 = "7d"
    MONTH_1 = "30d"
    MONTH_3 = "90d"

class PredictionRequest(BaseModel):
    symbol: str
    modelType: Optional[ModelType] = ModelType.AUTO
    horizon: Optional[int] = 5
    includeMetadata: Optional[bool] = False
    confidenceIntervals: Optional[bool] = True

class TrainingRequest(BaseModel):
    symbol: str
    modelType: ModelType
    options: Optional[Dict[str, Any]] = None
    validateResults: Optional[bool] = True

class RAGQueryRequest(BaseModel):
    query: str
    context: Optional[str] = None
    timeFrame: Optional[TimeFrame] = TimeFrame.WEEK_1
    relatedSymbols: Optional[List[str]] = None

class SentimentRequest(BaseModel):
    symbol: str
    timeFrame: Optional[TimeFrame] = TimeFrame.WEEK_1
    includeNews: Optional[bool] = True
    includeSocial: Optional[bool] = True
    includeAnalystRatings: Optional[bool] = True

class ModelSelectionCriteria(BaseModel):
    metric: Optional[str] = "rmse"  # rmse, mape, mae
    volatilityWeight: Optional[float] = 0.3
    trendStrengthWeight: Optional[float] = 0.3
    seasonalityWeight: Optional[float] = 0.2
    recentPerformanceWeight: Optional[float] = 0.2

class ModelPerformance(BaseModel):
    modelType: ModelType
    rmse: float
    mape: float
    mae: float
    lastUpdated: datetime.datetime
    performance_score: float
    suitable_conditions: List[str]

# ----- API Routes -----

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "service": "ml-service",
        "version": "2.0.0",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Generate predictions for a symbol using optimal or specified model"""
    try:
        # If AUTO, determine the best model for this symbol and market conditions
        if request.modelType == ModelType.AUTO:
            best_model = await auto_model_selector.select_optimal_model(
                symbol=request.symbol,
                horizon=request.horizon
            )
            logger.info(f"Auto-selected model for {request.symbol}: {best_model}")
            model_type = best_model
        else:
            model_type = request.modelType

        # Generate predictions
        result = prediction_pipeline.generate_predictions(
            symbol=request.symbol,
            horizon=request.horizon,
            model_type=model_type,
            include_confidence_intervals=request.confidenceIntervals
        )
        
        # Include model metadata if requested
        if request.includeMetadata:
            metadata = None
            
            # Get latest model metadata
            model_metadata = model_registry.get_models_by_symbol(request.symbol)
            for model in model_metadata:
                if model["modelType"].lower() == model_type.lower():
                    metadata = model
                    break
            
            if metadata:
                result["metadata"] = metadata
                
                # Include selection rationale if AUTO was used
                if request.modelType == ModelType.AUTO:
                    result["selectionRationale"] = auto_model_selector.get_selection_rationale(
                        symbol=request.symbol,
                        selected_model=model_type
                    )
        
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
            options=request.options,
            validate=request.validateResults
        )
        
        return {
            "status": "training_started",
            "message": f"Model training for {request.symbol} with {request.modelType} started in background",
            "symbol": request.symbol,
            "modelType": request.modelType,
            "estimatedCompletionTime": datetime.datetime.now() + datetime.timedelta(minutes=5)
        }
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get all available models"""
    try:
        return model_registry.get_all_models()
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
async def compare_models(symbol: str, criteria: ModelSelectionCriteria = Depends()):
    """Compare models for a symbol with customizable criteria"""
    try:
        comparison = auto_model_selector.compare_models(
            symbol=symbol,
            criteria=criteria.dict()
        )
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

@app.post("/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment for a symbol with configurable timeframe and sources"""
    try:
        result = await sentiment_analyzer.analyze(
            symbol=request.symbol,
            timeframe=request.timeFrame,
            include_news=request.includeNews,
            include_social=request.includeSocial,
            include_analyst_ratings=request.includeAnalystRatings
        )
        return result
    except Exception as e:
        logger.error(f"Sentiment analysis error for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sentiment/{symbol}/detailed")
async def get_detailed_sentiment(
    symbol: str, 
    timeframe: TimeFrame = TimeFrame.WEEK_1
):
    """Get detailed sentiment analysis for a symbol"""
    try:
        result = await sentiment_analyzer.get_detailed_sentiment(
            symbol=symbol,
            timeframe=timeframe
        )
        return result
    except Exception as e:
        logger.error(f"Detailed sentiment analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query")
async def query_rag(request: RAGQueryRequest):
    """Query the RAG system with enhanced context awareness"""
    try:
        # Get market context based on timeframe
        market_context = await data_processor.get_market_context(
            timeframe=request.timeFrame,
            related_symbols=request.relatedSymbols
        )
        
        # Combine with any user-provided context
        full_context = market_context
        if request.context:
            full_context = f"{request.context}\n\n{market_context}"
            
        # Process the RAG query with the enriched context
        response = {
            "query": request.query,
            "answer": f"This is a simulated response to: {request.query}",
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
            "contextTimeframe": request.timeFrame,
            "generatedAt": datetime.datetime.now().isoformat(),
            "relatedSymbols": request.relatedSymbols or 
                             (["AAPL", "MSFT", "GOOGL"] if "tech" in request.query.lower() else ["JPM", "BAC", "GS"])
        }
        
        # Add AI insights based on the query content and market data
        response["aiInsights"] = await generate_ai_insights(request.query, response["relatedSymbols"])
        
        return response
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/selector/{symbol}")
async def get_model_selector(
    symbol: str,
    horizon: int = 5,
    criteria: ModelSelectionCriteria = Depends()
):
    """Get AI-driven model recommendation for a symbol with customizable criteria"""
    try:
        recommended_model = await auto_model_selector.select_optimal_model(
            symbol=symbol,
            horizon=horizon,
            criteria=criteria.dict()
        )
        
        # Get selection rationale
        rationale = auto_model_selector.get_selection_rationale(
            symbol=symbol,
            selected_model=recommended_model
        )
        
        # Get performance metrics for all models
        all_models_performance = await auto_model_selector.get_all_models_performance(symbol)
        
        return {
            "recommendedModel": recommended_model,
            "selectionRationale": rationale,
            "modelsPerformance": all_models_performance,
            "marketConditions": {
                "volatility": "high" if "volatility" in rationale.lower() else "moderate",
                "trend": "strong" if "trend" in rationale.lower() else "weak",
                "seasonality": "present" if "seasonal" in rationale.lower() else "minimal"
            }
        }
    except Exception as e:
        logger.error(f"Error getting model recommendation for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# New feature: Portfolio Optimization endpoint
class OptimizationObjective(str, Enum):
    RETURN = "return"
    RISK = "risk"
    SHARPE = "sharpe"

class PortfolioOptimizationRequest(BaseModel):
    symbols: List[str]
    initialInvestment: float
    riskTolerance: float = Field(0.5, ge=0.0, le=1.0)
    objective: OptimizationObjective = OptimizationObjective.SHARPE
    constraints: Optional[Dict[str, Any]] = None

@app.post("/portfolio/optimize")
async def optimize_portfolio(request: PortfolioOptimizationRequest):
    """Optimize portfolio allocation using multiple ML models"""
    try:
        # Placeholder implementation - would be implemented with real portfolio optimization
        symbols_with_weights = []
        total_weight = 0.0
        
        # Simulated optimization result
        for symbol in request.symbols:
            # Use the best model for each symbol to predict expected returns
            model = await auto_model_selector.select_optimal_model(symbol)
            
            # Add to portfolio with a weight (this would be calculated by optimization algorithm)
            weight = 1.0 / len(request.symbols)  # Equal weight for now
            symbols_with_weights.append({
                "symbol": symbol,
                "weight": weight,
                "predictedReturn": 0.05 + (hash(symbol) % 10) / 100,  # Simulated return
                "risk": 0.02 + (hash(symbol) % 15) / 100,  # Simulated risk
                "modelUsed": model
            })
            total_weight += weight
        
        # Normalize weights to sum to 1.0
        for item in symbols_with_weights:
            item["weight"] = item["weight"] / total_weight
            
        return {
            "portfolio": symbols_with_weights,
            "expectedReturn": 0.08,  # Simulated overall return
            "expectedRisk": 0.04,  # Simulated overall risk
            "sharpeRatio": 2.0,  # Simulated Sharpe ratio
            "methodology": f"Portfolio optimized for {request.objective} with risk tolerance {request.riskTolerance}"
        }
    except Exception as e:
        logger.error(f"Portfolio optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function for generating AI insights
async def generate_ai_insights(query: str, symbols: List[str]) -> str:
    """Generate AI insights based on query and symbols (placeholder)"""
    # This would be implemented with a real LLM or fine-tuned model
    return f"Based on current market conditions and historical data for {', '.join(symbols)}, investors should consider the impact of recent economic indicators and sector performance trends."

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True) 