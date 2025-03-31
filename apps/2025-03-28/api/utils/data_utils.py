import pandas as pd
import numpy as np
from typing import Dict, Optional
import talib

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the given market data."""
    if len(data) < 30:
        raise ValueError("Insufficient data for calculating indicators")
        
    indicators = pd.DataFrame(index=data.index)
    
    # RSI
    indicators['rsi'] = talib.RSI(data['close'].values, timeperiod=14)
    
    # MACD
    macd, signal, _ = talib.MACD(
        data['close'].values,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )
    indicators['macd'] = macd
    indicators['macd_signal'] = signal
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(
        data['close'].values,
        timeperiod=20,
        nbdevup=2,
        nbdevdn=2,
        matype=0
    )
    indicators['bollinger_upper'] = upper
    indicators['bollinger_middle'] = middle
    indicators['bollinger_lower'] = lower
    
    # Moving Averages
    indicators['sma_20'] = talib.SMA(data['close'].values, timeperiod=20)
    indicators['sma_50'] = talib.SMA(data['close'].values, timeperiod=50)
    indicators['sma_200'] = talib.SMA(data['close'].values, timeperiod=200)
    
    # Volume Indicators
    indicators['obv'] = talib.OBV(data['close'].values, data['volume'].values)
    
    # Fill NaN values with forward fill then backward fill
    indicators = indicators.fillna(method='ffill').fillna(method='bfill')
    
    return indicators

def normalize_data(data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
    """Normalize the data using specified method."""
    if data.empty:
        raise ValueError("Empty DataFrame provided")
        
    normalized = data.copy()
    
    if method == 'minmax':
        for column in data.select_dtypes(include=[np.number]).columns:
            min_val = data[column].min()
            max_val = data[column].max()
            if max_val > min_val:
                normalized[column] = (data[column] - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        for column in data.select_dtypes(include=[np.number]).columns:
            mean = data[column].mean()
            std = data[column].std()
            if std > 0:
                normalized[column] = (data[column] - mean) / std
    
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return normalized

def validate_data(data: pd.DataFrame) -> bool:
    """Validate market data for completeness and correctness."""
    if data.empty:
        return False
        
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_columns):
        return False
    
    # Check for missing values
    if data[required_columns].isnull().any().any():
        return False
    
    # Check data types
    if not all(data[col].dtype in [np.float64, np.int64] for col in required_columns[:-1]):
        return False
    if not data['volume'].dtype in [np.int64, np.float64]:
        return False
    
    # Check logical constraints
    if not all(data['high'] >= data[['open', 'close', 'low']].max(axis=1)):
        return False
    if not all(data['low'] <= data[['open', 'close', 'high']].min(axis=1)):
        return False
    if not all(data['volume'] >= 0):
        return False
    
    return True

def prepare_model_data(
    data: pd.DataFrame,
    target_column: str = 'close',
    sequence_length: int = 60,
    prediction_length: int = 30,
    train_split: float = 0.8,
    include_indicators: bool = True
) -> Dict[str, np.ndarray]:
    """Prepare data for model training and prediction."""
    if len(data) < sequence_length + prediction_length:
        raise ValueError("Insufficient data for the specified sequence and prediction length")
    
    # Calculate indicators if requested
    if include_indicators:
        indicators = calculate_technical_indicators(data)
        data = pd.concat([data, indicators], axis=1)
    
    # Create sequences
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length - prediction_length + 1):
        sequence = data.iloc[i:(i + sequence_length)]
        target = data[target_column].iloc[(i + sequence_length):(i + sequence_length + prediction_length)]
        
        sequences.append(sequence.values)
        targets.append(target.values)
    
    X = np.array(sequences)
    y = np.array(targets)
    
    # Split into train and test
    train_size = int(len(X) * train_split)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_columns': data.columns.tolist()
    } 