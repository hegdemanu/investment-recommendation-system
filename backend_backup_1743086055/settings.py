"""
Core settings module for the Investment Recommendation System.

This file contains central configuration settings for various components of the system.
"""
import os
from pathlib import Path

# Base directory determination
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Dashboard Settings
DASHBOARD_SETTINGS = {
    # UI/UX Configuration
    "theme": "light",  # Options: light, dark
    "refresh_interval": 3600,  # In seconds (1 hour)
    "default_timeframe": "1M",  # Options: 1D, 1W, 1M, 3M, 6M, 1Y, YTD, MAX
    "max_stocks_displayed": 10,
    "enable_notifications": True,
    "notification_settings": {
        "price_alerts": True,
        "news_alerts": True,
        "prediction_alerts": True
    },
    
    # Performance Settings
    "cache_timeout": 900,  # In seconds (15 minutes)
    
    # Chart Settings
    "chart_settings": {
        "default_chart_type": "candlestick",  # Options: line, candlestick, ohlc
        "show_volume": True,
        "show_indicators": True,
        "default_indicators": ["SMA", "EMA", "RSI", "MACD"],
        "color_scheme": {
            "up": "#26a69a",      # Green for upward movement
            "down": "#ef5350",    # Red for downward movement
            "neutral": "#9e9e9e"  # Grey for neutral
        }
    },
    
    # Data Sources
    "data_sources": {
        "stocks": ["yfinance", "alpha_vantage"],
        "news": ["newsapi", "twitter", "reuters"],
        "fundamentals": ["yahoo_finance", "financial_modeling_prep"]
    }
}

# Directory Structure Settings
DASHBOARD_DIR = os.path.join(RESULTS_DIR, "dashboard")
DASHBOARD_FILE = os.path.join(DASHBOARD_DIR, "dashboard.html")
DASHBOARD_JSON_DIR = os.path.join(DASHBOARD_DIR, "json")
TRAINING_DIR = os.path.join(RESULTS_DIR, "training")
VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure directories exist
for dir_path in [RESULTS_DIR, DASHBOARD_DIR, DASHBOARD_JSON_DIR, TRAINING_DIR, 
                VALIDATION_DIR, REPORTS_DIR, MODELS_DIR, DATA_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Default stocks to display on dashboard
DEFAULT_STOCKS = [
    "RELIANCE.NS",  # Reliance Industries
    "TCS.NS",       # Tata Consultancy Services
    "HDFCBANK.NS",  # HDFC Bank
    "INFY.NS",      # Infosys
    "ICICIBANK.NS", # ICICI Bank
    "ITC.NS",       # ITC Limited
    "BAJFINANCE.NS",# Bajaj Finance
    "SBIN.NS",      # State Bank of India
    "AXISBANK.NS",  # Axis Bank
    "LT.NS"         # Larsen & Toubro
]

# Database Configuration
DB_CONFIG = {
    "engine": "sqlite",  # Options: sqlite, postgresql, mysql
    "name": os.path.join(BASE_DIR, "db", "investment_system.db"),
    "user": "",
    "password": "",
    "host": "",
    "port": ""
}

# API Configuration
API_CONFIG = {
    "news_api_key": os.environ.get("NEWS_API_KEY", ""),
    "alpha_vantage_api_key": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
    "fmp_api_key": os.environ.get("FMP_API_KEY", "")
}

# File Paths
LOG_FILE = os.path.join(BASE_DIR, "logs", "app.log")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOG_FILE,
    "max_size": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5
}

# Model Configuration
MODEL_CONFIG = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "test_split": 0.1,
    "window_size": 30,  # For time series models
    "prediction_horizon": 7  # How many days into the future to predict
} 