import pytest
import numpy as np
from app.models.training_pipeline import ModelTrainingPipeline
from app.models.prediction_pipeline import PredictionPipeline
from app.models.sentiment_pipeline import SentimentAnalyzer

@pytest.fixture
def training_pipeline():
    return ModelTrainingPipeline()

@pytest.fixture
def prediction_pipeline():
    return PredictionPipeline()

@pytest.fixture
def sentiment_analyzer():
    return SentimentAnalyzer()

def test_training_pipeline_initialization(training_pipeline):
    assert training_pipeline is not None
    assert hasattr(training_pipeline, 'lstm_model')
    assert hasattr(training_pipeline, 'arima_model')
    assert hasattr(training_pipeline, 'prophet_model')

def test_prediction_pipeline_initialization(prediction_pipeline):
    assert prediction_pipeline is not None
    assert hasattr(prediction_pipeline, 'models')
    assert hasattr(prediction_pipeline, 'cache')

def test_sentiment_analyzer_initialization(sentiment_analyzer):
    assert sentiment_analyzer is not None
    assert hasattr(sentiment_analyzer, 'model')
    assert hasattr(sentiment_analyzer, 'tokenizer')

@pytest.mark.asyncio
async def test_sentiment_analysis(sentiment_analyzer):
    text = "Apple's new iPhone sales exceed expectations"
    result = await sentiment_analyzer.analyze_text(text)
    assert isinstance(result, dict)
    assert "sentiment" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert isinstance(result["confidence"], float)
    assert 0 <= result["confidence"] <= 1

@pytest.mark.asyncio
async def test_prediction_generation(prediction_pipeline):
    symbol = "AAPL"
    days = 30
    predictions = await prediction_pipeline.generate_predictions(
        symbol=symbol,
        horizon=days
    )
    assert isinstance(predictions, dict)
    assert "predictions" in predictions
    assert "confidence" in predictions
    assert len(predictions["predictions"]) == days

def test_model_training(training_pipeline):
    # Create sample data
    dates = np.arange('2023-01-01', '2024-01-01', dtype='datetime64[D]')
    prices = np.random.normal(100, 10, len(dates))
    data = {
        "dates": dates,
        "prices": prices
    }
    
    # Test training
    results = training_pipeline.train_all_models(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    
    assert isinstance(results, dict)
    assert "lstm" in results
    assert "arima" in results
    assert "prophet" in results
    assert all(isinstance(score, float) for score in results.values()) 