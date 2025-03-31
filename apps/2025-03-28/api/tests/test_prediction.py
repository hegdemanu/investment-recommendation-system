import pytest
import pandas as pd
import numpy as np
from app.models.prediction_pipeline import PredictionPipeline
from app.models.sentiment_pipeline import SentimentAnalyzer

@pytest.fixture
def prediction_pipeline():
    return PredictionPipeline()

@pytest.fixture
def sentiment_analyzer():
    return SentimentAnalyzer()

@pytest.fixture
def sample_historical_data():
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    return pd.DataFrame({
        'date': dates,
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(105, 10, len(dates)),
        'low': np.random.normal(95, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })

@pytest.mark.asyncio
async def test_prediction_generation(prediction_pipeline, sample_historical_data):
    symbol = "AAPL"
    days = 30
    predictions = await prediction_pipeline.generate_predictions(
        symbol=symbol,
        horizon=days,
        historical_data=sample_historical_data
    )
    
    assert isinstance(predictions, dict)
    assert "predictions" in predictions
    assert "confidence" in predictions
    assert len(predictions["predictions"]) == days
    assert all(isinstance(pred, float) for pred in predictions["predictions"])
    assert 0 <= predictions["confidence"] <= 1

@pytest.mark.asyncio
async def test_prediction_with_sentiment(
    prediction_pipeline,
    sentiment_analyzer,
    sample_historical_data
):
    symbol = "AAPL"
    days = 30
    predictions = await prediction_pipeline.generate_predictions(
        symbol=symbol,
        horizon=days,
        historical_data=sample_historical_data,
        sentiment_analyzer=sentiment_analyzer
    )
    
    assert isinstance(predictions, dict)
    assert "predictions" in predictions
    assert "confidence" in predictions
    assert "sentiment" in predictions
    assert len(predictions["predictions"]) == days
    assert 0 <= predictions["confidence"] <= 1
    assert predictions["sentiment"] in ["positive", "negative", "neutral"]

@pytest.mark.asyncio
async def test_prediction_caching(prediction_pipeline, sample_historical_data):
    symbol = "AAPL"
    days = 30
    
    # First prediction
    predictions1 = await prediction_pipeline.generate_predictions(
        symbol=symbol,
        horizon=days,
        historical_data=sample_historical_data
    )
    
    # Second prediction (should use cache)
    predictions2 = await prediction_pipeline.generate_predictions(
        symbol=symbol,
        horizon=days,
        historical_data=sample_historical_data
    )
    
    assert predictions1 == predictions2

@pytest.mark.asyncio
async def test_prediction_error_handling(prediction_pipeline):
    # Test with invalid symbol
    with pytest.raises(ValueError):
        await prediction_pipeline.generate_predictions(
            symbol="",
            horizon=30
        )
    
    # Test with invalid horizon
    with pytest.raises(ValueError):
        await prediction_pipeline.generate_predictions(
            symbol="AAPL",
            horizon=0
        )
    
    # Test with invalid historical data
    with pytest.raises(ValueError):
        await prediction_pipeline.generate_predictions(
            symbol="AAPL",
            horizon=30,
            historical_data=pd.DataFrame()
        )

@pytest.mark.asyncio
async def test_prediction_with_different_horizons(prediction_pipeline, sample_historical_data):
    symbol = "AAPL"
    horizons = [7, 14, 30]
    
    for days in horizons:
        predictions = await prediction_pipeline.generate_predictions(
            symbol=symbol,
            horizon=days,
            historical_data=sample_historical_data
        )
        
        assert isinstance(predictions, dict)
        assert "predictions" in predictions
        assert len(predictions["predictions"]) == days
        assert all(isinstance(pred, float) for pred in predictions["predictions"])

@pytest.mark.asyncio
async def test_prediction_confidence_calculation(prediction_pipeline, sample_historical_data):
    symbol = "AAPL"
    days = 30
    predictions = await prediction_pipeline.generate_predictions(
        symbol=symbol,
        horizon=days,
        historical_data=sample_historical_data
    )
    
    # Test confidence calculation
    assert isinstance(predictions["confidence"], float)
    assert 0 <= predictions["confidence"] <= 1
    
    # Test confidence decreases with horizon
    short_predictions = await prediction_pipeline.generate_predictions(
        symbol=symbol,
        horizon=7,
        historical_data=sample_historical_data
    )
    assert short_predictions["confidence"] >= predictions["confidence"]

@pytest.mark.asyncio
async def test_prediction_with_technical_indicators(prediction_pipeline, sample_historical_data):
    symbol = "AAPL"
    days = 30
    predictions = await prediction_pipeline.generate_predictions(
        symbol=symbol,
        horizon=days,
        historical_data=sample_historical_data,
        include_technical=True
    )
    
    assert isinstance(predictions, dict)
    assert "predictions" in predictions
    assert "technical_indicators" in predictions
    assert "rsi" in predictions["technical_indicators"]
    assert "macd" in predictions["technical_indicators"]
    assert "bollinger" in predictions["technical_indicators"] 