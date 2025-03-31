import pytest
import pandas as pd
import numpy as np
from app.models.training_pipeline import ModelTrainingPipeline

@pytest.fixture
def training_pipeline():
    return ModelTrainingPipeline()

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    return pd.DataFrame({
        'date': dates,
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(105, 10, len(dates)),
        'low': np.random.normal(95, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })

def test_training_pipeline_initialization(training_pipeline):
    assert training_pipeline is not None
    assert hasattr(training_pipeline, 'lstm_model')
    assert hasattr(training_pipeline, 'arima_model')
    assert hasattr(training_pipeline, 'prophet_model')

def test_data_preparation(training_pipeline, sample_data):
    prepared_data = training_pipeline.prepare_data(sample_data)
    assert isinstance(prepared_data, dict)
    assert "lstm" in prepared_data
    assert "arima" in prepared_data
    assert "prophet" in prepared_data

def test_lstm_model_training(training_pipeline, sample_data):
    prepared_data = training_pipeline.prepare_data(sample_data)
    model, history = training_pipeline.train_lstm(prepared_data["lstm"])
    
    assert model is not None
    assert history is not None
    assert "loss" in history.history
    assert "val_loss" in history.history

def test_arima_model_training(training_pipeline, sample_data):
    prepared_data = training_pipeline.prepare_data(sample_data)
    model = training_pipeline.train_arima(prepared_data["arima"])
    
    assert model is not None
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')

def test_prophet_model_training(training_pipeline, sample_data):
    prepared_data = training_pipeline.prepare_data(sample_data)
    model = training_pipeline.train_prophet(prepared_data["prophet"])
    
    assert model is not None
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')

def test_model_saving_and_loading(training_pipeline, sample_data, tmp_path):
    # Train models
    prepared_data = training_pipeline.prepare_data(sample_data)
    lstm_model, _ = training_pipeline.train_lstm(prepared_data["lstm"])
    arima_model = training_pipeline.train_arima(prepared_data["arima"])
    prophet_model = training_pipeline.train_prophet(prepared_data["prophet"])
    
    # Save models
    model_dir = tmp_path / "models"
    training_pipeline.save_models(
        lstm_model=lstm_model,
        arima_model=arima_model,
        prophet_model=prophet_model,
        model_dir=model_dir
    )
    
    # Load models
    loaded_models = training_pipeline.load_models(model_dir)
    assert "lstm" in loaded_models
    assert "arima" in loaded_models
    assert "prophet" in loaded_models

def test_model_evaluation(training_pipeline, sample_data):
    # Train models
    prepared_data = training_pipeline.prepare_data(sample_data)
    lstm_model, _ = training_pipeline.train_lstm(prepared_data["lstm"])
    arima_model = training_pipeline.train_arima(prepared_data["arima"])
    prophet_model = training_pipeline.train_prophet(prepared_data["prophet"])
    
    # Evaluate models
    metrics = training_pipeline.evaluate_models(
        lstm_model=lstm_model,
        arima_model=arima_model,
        prophet_model=prophet_model,
        test_data=prepared_data
    )
    
    assert isinstance(metrics, dict)
    assert "lstm" in metrics
    assert "arima" in metrics
    assert "prophet" in metrics
    assert all(isinstance(score, float) for score in metrics.values())

def test_model_training_with_different_horizons(training_pipeline, sample_data):
    horizons = [7, 14, 30]
    
    for horizon in horizons:
        prepared_data = training_pipeline.prepare_data(sample_data, horizon=horizon)
        lstm_model, _ = training_pipeline.train_lstm(prepared_data["lstm"])
        arima_model = training_pipeline.train_arima(prepared_data["arima"])
        prophet_model = training_pipeline.train_prophet(prepared_data["prophet"])
        
        assert lstm_model is not None
        assert arima_model is not None
        assert prophet_model is not None

def test_model_training_error_handling(training_pipeline):
    # Test with empty data
    with pytest.raises(ValueError):
        training_pipeline.prepare_data(pd.DataFrame())
    
    # Test with invalid data
    invalid_data = pd.DataFrame({
        'date': ['2023-01-01'],
        'price': ['not_a_number']
    })
    with pytest.raises(ValueError):
        training_pipeline.prepare_data(invalid_data)
    
    # Test with insufficient data
    small_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'price': [100 + i for i in range(5)]
    })
    with pytest.raises(ValueError):
        training_pipeline.prepare_data(small_data)

def test_model_training_with_validation_split(training_pipeline, sample_data):
    prepared_data = training_pipeline.prepare_data(
        sample_data,
        validation_split=0.2
    )
    
    assert "train" in prepared_data["lstm"]
    assert "val" in prepared_data["lstm"]
    assert len(prepared_data["lstm"]["train"]) > len(prepared_data["lstm"]["val"]) 