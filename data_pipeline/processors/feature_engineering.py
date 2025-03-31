import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import talib


class FeatureEngineering:
    """
    Feature engineering for market data
    """
    
    @staticmethod
    def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()
        
        # Convert column names if they are not in the expected format
        # Sometimes Alpha Vantage returns differently named columns
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume',
            'adjusted close': 'Adjusted close',
            '5. adjusted close': 'Adjusted close'
        }
        
        df = df.rename(columns={col: column_mapping.get(col.lower(), col) for col in df.columns})
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")
            
        # Moving Averages
        df['MA5'] = talib.SMA(df['Close'], timeperiod=5)
        df['MA10'] = talib.SMA(df['Close'], timeperiod=10)
        df['MA20'] = talib.SMA(df['Close'], timeperiod=20)
        df['MA50'] = talib.SMA(df['Close'], timeperiod=50)
        df['MA200'] = talib.SMA(df['Close'], timeperiod=200)
        
        # Exponential Moving Averages
        df['EMA5'] = talib.EMA(df['Close'], timeperiod=5)
        df['EMA10'] = talib.EMA(df['Close'], timeperiod=10)
        df['EMA20'] = talib.EMA(df['Close'], timeperiod=20)
        
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
            df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # Relative Strength Index
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        
        # Moving Average Convergence Divergence
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # Commodity Channel Index
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Average Directional Index
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Rate of Change
        df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
        
        # Williams %R
        df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Stochastic Oscillator
        df['SLOWK'], df['SLOWD'] = talib.STOCH(
            df['High'], df['Low'], df['Close'], 
            fastk_period=5, slowk_period=3, slowk_matype=0, 
            slowd_period=3, slowd_matype=0
        )
        
        # On-Balance Volume
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        # Average True Range
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Chaikin A/D Line
        df['AD'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Price-Volume Trend
        df['PVT'] = (df['Close'].pct_change() * df['Volume']).cumsum()
        
        # Percentage Price Oscillator
        df['PPO'] = talib.PPO(df['Close'], fastperiod=12, slowperiod=26, matype=0)
        
        # Returns
        df['daily_return'] = df['Close'].pct_change()
        df['5d_return'] = df['Close'].pct_change(periods=5)
        df['10d_return'] = df['Close'].pct_change(periods=10)
        df['20d_return'] = df['Close'].pct_change(periods=20)
        
        # Volatility
        df['volatility_5d'] = df['daily_return'].rolling(window=5).std()
        df['volatility_10d'] = df['daily_return'].rolling(window=10).std()
        df['volatility_20d'] = df['daily_return'].rolling(window=20).std()
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    @staticmethod
    def prepare_model_data(data: pd.DataFrame, lookback_days: int = 60, prediction_days: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training
        
        Args:
            data: DataFrame with features
            lookback_days: Number of past days to use for prediction
            prediction_days: Number of future days to predict
            
        Returns:
            X: Input data
            y: Target data
        """
        # Select features
        feature_columns = [col for col in data.columns if col != 'date']
        
        # Normalize data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[feature_columns])
        
        X = []
        y = []
        
        for i in range(lookback_days, len(scaled_data) - prediction_days):
            X.append(scaled_data[i - lookback_days:i])
            y.append(scaled_data[i + prediction_days - 1, data.columns.get_loc('Close')])
            
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    @staticmethod
    def feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
        """
        Calculate feature importance for a trained model
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        # Get feature importance
        importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df 