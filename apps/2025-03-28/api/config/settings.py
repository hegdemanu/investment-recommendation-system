#!/usr/bin/env python3
"""
Configuration Settings for Investment Recommendation System
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist"""
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# API Keys and external services
API_KEYS = {
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_KEY"),
    "news_api": os.getenv("NEWS_API_KEY"),
    "finnhub": os.getenv("FINNHUB_API_KEY")
}

# Model configurations
MODEL_CONFIG = {
    "lstm": {
        "units": 50,
        "dropout": 0.2,
        "epochs": 100,
        "batch_size": 32,
        "sequence_length": 60
    },
    "arima": {
        "order": (5,1,2),
        "seasonal_order": (1,1,1,12)
    },
    "prophet": {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10,
        "seasonality_mode": "multiplicative"
    }
}

# Data pipeline settings
DATA_CONFIG = {
    "cache_expiry": 24,  # hours
    "batch_size": 5,
    "retry_attempts": 3,
    "retry_delay": 5,  # seconds
    "sources": ["yfinance", "alpha_vantage"]
}

# API configurations
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", 8000)),
    "debug": os.getenv("DEBUG", "False").lower() == "true",
    "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    "rate_limit": {
        "requests": int(os.getenv("RATE_LIMIT_REQUESTS", 100)),
        "window": int(os.getenv("RATE_LIMIT_WINDOW", 3600))
    }
}

# Sentiment analysis settings
SENTIMENT_CONFIG = {
    "model_name": "ProsusAI/finbert",
    "batch_size": 16,
    "max_length": 512,
    "weights": {
        "negative": -1.0,
        "neutral": 0.0,
        "positive": 1.0
    }
}

# Database settings
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "investment_system"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "pool_size": int(os.getenv("DB_POOL_SIZE", 5))
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler"
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR / "app.log",
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}

# Cache settings
CACHE_CONFIG = {
    "type": os.getenv("CACHE_TYPE", "redis"),
    "url": os.getenv("CACHE_URL", "redis://localhost:6379/0"),
    "ttl": int(os.getenv("CACHE_TTL", 3600))
}

# Model serving settings
MODEL_SERVING = {
    "max_batch_size": int(os.getenv("MODEL_MAX_BATCH", 32)),
    "timeout": int(os.getenv("MODEL_TIMEOUT", 30)),
    "max_queue_size": int(os.getenv("MODEL_QUEUE_SIZE", 100))
}

# Feature flags
FEATURES = {
    "use_sentiment": os.getenv("USE_SENTIMENT", "True").lower() == "true",
    "use_news": os.getenv("USE_NEWS", "True").lower() == "true",
    "use_technical": os.getenv("USE_TECHNICAL", "True").lower() == "true",
    "enable_caching": os.getenv("ENABLE_CACHING", "True").lower() == "true"
}

# Dashboard settings
DASHBOARD_CONFIG = {
    "update_interval": int(os.getenv("DASHBOARD_UPDATE_INTERVAL", 300)),
    "max_points": int(os.getenv("DASHBOARD_MAX_POINTS", 1000)),
    "default_timeframe": os.getenv("DASHBOARD_TIMEFRAME", "1M")
}

# Environment-specific settings
ENV = os.getenv("ENV", "development")

if ENV == "development":
    DEBUG = True
    TESTING = True
elif ENV == "production":
    DEBUG = False
    TESTING = False
else:
    raise ValueError(f"Invalid environment: {ENV}")

# Export all settings
__all__ = [
    "API_KEYS",
    "MODEL_CONFIG",
    "DATA_CONFIG",
    "API_CONFIG",
    "SENTIMENT_CONFIG",
    "DB_CONFIG",
    "LOGGING_CONFIG",
    "CACHE_CONFIG",
    "MODEL_SERVING",
    "FEATURES",
    "DASHBOARD_CONFIG",
    "ENV",
    "DEBUG",
    "TESTING",
    "ensure_directories"
] 