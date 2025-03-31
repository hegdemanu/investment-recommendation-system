import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import ta
from datetime import datetime, timedelta
import json
import os

class DataProcessor:
    def __init__(self):
        self.data_cache = {}
        self.cache_duration = timedelta(minutes=15)  # Cache data for 15 minutes
        
    def get_stock_data(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data with technical indicators
        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        try:
            cache_key = f"{symbol}_{period}_{interval}"
            
            # Check cache
            if cache_key in self.data_cache:
                timestamp, data = self.data_cache[cache_key]
                if datetime.now() - timestamp < self.cache_duration:
                    return data
            
            # Fetch new data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Calculate technical indicators
            data = self.add_technical_indicators(data)
            
            # Update cache
            self.data_cache[cache_key] = (datetime.now(), data)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching stock data: {str(e)}")
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        df = data.copy()
        
        # Trend indicators
        df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['macd'] = ta.trend.MACD(df['Close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['Close']).macd_signal()
        df['macd_diff'] = ta.trend.MACD(df['Close']).macd_diff()
        
        # Momentum indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['stoch'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
        df['stoch_signal'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Volume indicators
        df['volume_sma'] = ta.volume.SMAIndicator(df['Volume'], window=20).sma_indicator()
        df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        ).volume_weighted_average_price()
        
        # Price changes
        df['price_change'] = df['Close'].pct_change()
        df['volume_change'] = df['Volume'].pct_change()
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        return df
    
    def prepare_model_input(
        self,
        data: pd.DataFrame,
        lookback: int = 60,
        target_column: str = 'Close'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Prepare input sequences for model prediction"""
        # Select features
        feature_columns = [
            'Close', 'Volume', 'rsi', 'macd', 'macd_signal',
            'bb_high', 'bb_low', 'sma_20', 'sma_50'
        ]
        
        # Ensure all required columns exist
        missing_cols = [col for col in feature_columns if col not in data.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return np.array([]), None
        
        # Create feature matrix
        features = data[feature_columns].values
        
        # Normalize features
        features = self._normalize_features(features)
        
        # Create sequences
        X = []
        y = []
        
        for i in range(len(features) - lookback):
            X.append(features[i:(i + lookback)])
            if i + lookback < len(features):
                y.append(features[i + lookback, 0])  # Target is next day's close price
        
        if not X:
            return np.array([]), None
        
        return np.array(X), np.array(y) if y else None
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using min-max scaling"""
        # Apply scaling per feature
        normalized = np.zeros_like(features)
        
        for i in range(features.shape[1]):
            feature_min = np.min(features[:, i])
            feature_max = np.max(features[:, i])
            
            if feature_max - feature_min != 0:
                normalized[:, i] = (features[:, i] - feature_min) / (feature_max - feature_min)
            else:
                normalized[:, i] = features[:, i]
        
        return normalized
    
    def save_processed_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        data_dir: str = "processed_data"
    ) -> bool:
        """Save processed data to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Save data
            file_path = os.path.join(data_dir, f"{symbol}_processed.csv")
            data.to_csv(file_path)
            
            # Save metadata
            metadata = {
                "symbol": symbol,
                "processed_at": datetime.now().isoformat(),
                "rows": len(data),
                "columns": list(data.columns),
                "features_added": [
                    "technical_indicators",
                    "momentum_indicators",
                    "volatility_indicators",
                    "volume_indicators"
                ]
            }
            
            metadata_path = os.path.join(data_dir, f"{symbol}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving processed data: {str(e)}")
            return False
    
    def load_processed_data(
        self,
        symbol: str,
        data_dir: str = "processed_data"
    ) -> Optional[pd.DataFrame]:
        """Load processed data from disk"""
        try:
            file_path = os.path.join(data_dir, f"{symbol}_processed.csv")
            if not os.path.exists(file_path):
                return None
            
            data = pd.read_csv(file_path, index_col=0)
            data.index = pd.to_datetime(data.index)
            
            return data
            
        except Exception as e:
            print(f"Error loading processed data: {str(e)}")
            return None
    
    def get_market_data(self) -> Dict:
        """Get overall market data (S&P 500, VIX, etc.)"""
        try:
            market_symbols = ['^GSPC', '^VIX', '^DJI', '^IXIC']
            market_data = {}
            
            for symbol in market_symbols:
                data = self.get_stock_data(symbol, period="5d")
                latest = data.iloc[-1]
                
                market_data[symbol] = {
                    'price': float(latest['Close']),
                    'change': float(latest['Close'] / data.iloc[-2]['Close'] - 1),
                    'volume': float(latest['Volume']) if 'Volume' in latest else None
                }
            
            return market_data
            
        except Exception as e:
            raise Exception(f"Error fetching market data: {str(e)}")
    
    def get_sector_performance(self) -> Dict:
        """Get sector ETF performance"""
        try:
            sector_etfs = {
                'XLK': 'Technology',
                'XLF': 'Financial',
                'XLE': 'Energy',
                'XLV': 'Healthcare',
                'XLI': 'Industrial',
                'XLP': 'Consumer Staples',
                'XLY': 'Consumer Discretionary',
                'XLB': 'Materials',
                'XLU': 'Utilities',
                'XLRE': 'Real Estate'
            }
            
            performance = {}
            for symbol, sector in sector_etfs.items():
                data = self.get_stock_data(symbol, period="5d")
                latest = data.iloc[-1]
                
                performance[sector] = {
                    'symbol': symbol,
                    'price': float(latest['Close']),
                    'change': float(latest['Close'] / data.iloc[-2]['Close'] - 1)
                }
            
            return performance
            
        except Exception as e:
            raise Exception(f"Error fetching sector performance: {str(e)}")
    
    def get_correlation_matrix(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Calculate correlation matrix between multiple symbols"""
        try:
            # Fetch data for all symbols
            prices = pd.DataFrame()
            for symbol in symbols:
                data = self.get_stock_data(symbol, period=period)
                prices[symbol] = data['Close']
            
            # Calculate returns
            returns = prices.pct_change()
            
            # Calculate correlation matrix
            correlation = returns.corr()
            
            return correlation
            
        except Exception as e:
            raise Exception(f"Error calculating correlations: {str(e)}")
    
    def get_risk_metrics(self, symbol: str, period: str = "1y") -> Dict:
        """Calculate risk metrics for a symbol"""
        try:
            data = self.get_stock_data(symbol, period=period)
            returns = data['Close'].pct_change().dropna()
            
            metrics = {
                'volatility': float(returns.std() * np.sqrt(252)),  # Annualized volatility
                'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)),  # Annualized Sharpe ratio
                'max_drawdown': float(self._calculate_max_drawdown(data['Close'])),
                'var_95': float(np.percentile(returns, 5)),  # 95% Value at Risk
                'skewness': float(returns.skew()),
                'kurtosis': float(returns.kurtosis())
            }
            
            return metrics
            
        except Exception as e:
            raise Exception(f"Error calculating risk metrics: {str(e)}")
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from peak"""
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return abs(drawdown.min()) 