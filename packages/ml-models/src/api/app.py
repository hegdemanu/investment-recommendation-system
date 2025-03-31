from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stock_predictor import StockPredictor
from models.sentiment_analyzer import SentimentAnalyzer
from models.model_trainer import ModelTrainer
from utils.data_processor import DataProcessor
from utils.model_registry import ModelRegistry

app = FastAPI(title="Investment ML Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_registry = ModelRegistry()
stock_predictor = StockPredictor(model_registry)
sentiment_analyzer = SentimentAnalyzer(model_registry)
model_trainer = ModelTrainer(model_registry)
data_processor = DataProcessor()

class PredictionRequest(BaseModel):
    symbol: str
    features: List[float]
    timeframe: str = "1d"

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    sentiment: float
    technical_indicators: Dict[str, float]

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models": {
            "lstm": len(model_registry.models_info["lstm"]),
            "prophet": len(model_registry.models_info["prophet"]),
            "sentiment": len(model_registry.models_info["sentiment"])
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Check if model exists
        if not model_registry.get_model_metadata("lstm", request.symbol):
            raise HTTPException(status_code=404, detail=f"No trained model found for {request.symbol}")
        
        # Get prediction
        prediction, confidence = stock_predictor.predict(
            request.symbol,
            request.features
        )
        
        # Get sentiment
        sentiment = sentiment_analyzer.analyze(request.symbol)
        
        # Get technical indicators
        data = data_processor.get_stock_data(
            request.symbol,
            period="1mo",
            interval=request.timeframe
        )
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol")
            
        latest = data.iloc[-1]
        indicators = {
            "rsi": float(latest.get("rsi", 0)),
            "macd": float(latest.get("macd", 0)),
            "volume_sma": float(latest.get("volume_sma", 0)),
            "bb_high": float(latest.get("bb_high", 0)),
            "bb_low": float(latest.get("bb_low", 0))
        }
        
        return {
            "prediction": float(prediction),
            "confidence": float(confidence),
            "sentiment": float(sentiment),
            "technical_indicators": indicators
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/{symbol}")
async def train_model(symbol: str):
    try:
        success = await model_trainer.train(symbol)
        if success:
            return {"status": "success", "message": f"Model trained for {symbol}"}
        else:
            raise HTTPException(status_code=500, detail="Training failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models(model_type: Optional[str] = None):
    try:
        return model_registry.list_available_models(model_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/{model_type}/{symbol}")
async def delete_model(model_type: str, symbol: str):
    try:
        if model_registry.unregister_model(model_type, symbol):
            return {"status": "success", "message": f"Model {symbol} unregistered"}
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 