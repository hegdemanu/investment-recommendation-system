import pytest
import pandas as pd
import numpy as np
from app.data.market_data import MarketDataPipeline
from app.data.preprocessing import DataPreprocessor
from app.data.feature_engineering import FeatureEngineer

@pytest.fixture
def market_data_pipeline():
    return MarketDataPipeline()

@pytest.fixture
def data_preprocessor():
    return DataPreprocessor()

@pytest.fixture
def feature_engineer():
    return FeatureEngineer()

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.normal(100, 10, len(dates)),
        'high': np.random.normal(105, 10, len(dates)),
        'low': np.random.normal(95, 10, len(dates)),
        'close': np.random.normal(100, 10, len(dates)),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    return data

@pytest.mark.asyncio
async def test_market_data_fetching(market_data_pipeline):
    symbol = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    
    data = await market_data_pipeline.fetch_market_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_data_preprocessing(data_preprocessor, sample_data):
    processed_data = data_preprocessor.preprocess_data(sample_data)
    
    assert isinstance(processed_data, pd.DataFrame)
    assert not processed_data.empty
    assert not processed_data.isnull().any().any()
    assert all(col in processed_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

def test_feature_engineering(feature_engineer, sample_data):
    features = feature_engineer.create_features(sample_data)
    
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert 'rsi' in features.columns
    assert 'macd' in features.columns
    assert 'bollinger_upper' in features.columns
    assert 'bollinger_lower' in features.columns

def test_data_validation(sample_data):
    # Test for missing values
    assert not sample_data.isnull().any().any()
    
    # Test for data types
    assert sample_data['date'].dtype == 'datetime64[ns]'
    assert all(sample_data[col].dtype in ['float64', 'int64'] for col in ['open', 'high', 'low', 'close', 'volume'])
    
    # Test for logical constraints
    assert all(sample_data['high'] >= sample_data['low'])
    assert all(sample_data['high'] >= sample_data['open'])
    assert all(sample_data['high'] >= sample_data['close'])
    assert all(sample_data['low'] <= sample_data['open'])
    assert all(sample_data['low'] <= sample_data['close'])
    assert all(sample_data['volume'] >= 0)

def test_data_normalization(data_preprocessor, sample_data):
    normalized_data = data_preprocessor.normalize_data(sample_data)
    
    assert isinstance(normalized_data, pd.DataFrame)
    assert not normalized_data.empty
    assert all(col in normalized_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    assert all(normalized_data[col].between(-1, 1).all() for col in ['open', 'high', 'low', 'close']) 