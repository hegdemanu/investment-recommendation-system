import pytest
import os
import json
import pandas as pd
from app.utils.file_utils import (
    ensure_directory,
    save_json,
    load_json,
    save_dataframe,
    load_dataframe
)
from app.utils.data_utils import (
    calculate_technical_indicators,
    normalize_data,
    validate_data
)

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', end='2024-01-01', freq='D'),
        'price': [100 + i for i in range(366)],
        'volume': [1000000 + i * 1000 for i in range(366)]
    })

@pytest.fixture
def sample_json():
    return {
        'symbol': 'AAPL',
        'predictions': {
            'dates': ['2024-01-01', '2024-01-02'],
            'prices': [150.0, 151.0],
            'confidence': [0.8, 0.85]
        }
    }

def test_ensure_directory():
    test_dir = "test_dir"
    ensure_directory(test_dir)
    assert os.path.exists(test_dir)
    os.rmdir(test_dir)

def test_save_and_load_json(tmp_path, sample_json):
    file_path = tmp_path / "test.json"
    save_json(sample_json, file_path)
    loaded_data = load_json(file_path)
    assert loaded_data == sample_json

def test_save_and_load_dataframe(tmp_path, sample_dataframe):
    file_path = tmp_path / "test.csv"
    save_dataframe(sample_dataframe, file_path)
    loaded_df = load_dataframe(file_path)
    pd.testing.assert_frame_equal(loaded_df, sample_dataframe)

def test_calculate_technical_indicators(sample_dataframe):
    indicators = calculate_technical_indicators(sample_dataframe)
    assert isinstance(indicators, pd.DataFrame)
    assert 'rsi' in indicators.columns
    assert 'macd' in indicators.columns
    assert 'bollinger_upper' in indicators.columns
    assert 'bollinger_lower' in indicators.columns

def test_normalize_data(sample_dataframe):
    normalized_df = normalize_data(sample_dataframe)
    assert isinstance(normalized_df, pd.DataFrame)
    assert all(normalized_df[col].between(-1, 1).all() for col in ['price'])
    assert all(normalized_df[col].between(0, 1).all() for col in ['volume'])

def test_validate_data(sample_dataframe):
    # Test valid data
    assert validate_data(sample_dataframe)
    
    # Test invalid data (missing values)
    invalid_df = sample_dataframe.copy()
    invalid_df.loc[0, 'price'] = None
    assert not validate_data(invalid_df)
    
    # Test invalid data (negative values)
    invalid_df = sample_dataframe.copy()
    invalid_df.loc[0, 'volume'] = -1
    assert not validate_data(invalid_df)

def test_file_operations_with_invalid_paths():
    # Test with invalid file paths
    with pytest.raises(Exception):
        save_json({}, "/invalid/path/test.json")
    
    with pytest.raises(Exception):
        load_json("/invalid/path/test.json")
    
    with pytest.raises(Exception):
        save_dataframe(pd.DataFrame(), "/invalid/path/test.csv")
    
    with pytest.raises(Exception):
        load_dataframe("/invalid/path/test.csv")

def test_data_validation_edge_cases():
    # Test empty dataframe
    assert not validate_data(pd.DataFrame())
    
    # Test dataframe with wrong data types
    invalid_df = pd.DataFrame({
        'date': ['2023-01-01'],
        'price': ['not_a_number'],
        'volume': [1000000]
    })
    assert not validate_data(invalid_df)
    
    # Test dataframe with missing required columns
    invalid_df = pd.DataFrame({
        'date': ['2023-01-01'],
        'price': [100.0]
    })
    assert not validate_data(invalid_df)

def test_technical_indicators_edge_cases():
    # Test with empty dataframe
    with pytest.raises(ValueError):
        calculate_technical_indicators(pd.DataFrame())
    
    # Test with insufficient data
    small_df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'price': [100 + i for i in range(5)],
        'volume': [1000000 + i * 1000 for i in range(5)]
    })
    with pytest.raises(ValueError):
        calculate_technical_indicators(small_df) 