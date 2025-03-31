"""
Main configuration module for the investment recommendation system backend.

This module handles configuration loading and provides access to settings from
environment variables and configuration files.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# Import utility functions
from app.utils.config_utils import Config, load_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, CACHE_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv not installed. Skipping .env file loading.")
except Exception as e:
    logger.warning(f"Error loading .env file: {str(e)}")

# Load configuration
config_path = os.environ.get("CONFIG_PATH")
if config_path:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
else:
    # Default config path
    config_path = BASE_DIR / "config.yaml"
    if not config_path.exists():
        config_path = BASE_DIR / "config.json"
        if not config_path.exists():
            config_path = None

# Load settings
settings = load_settings(config_path=config_path, env_prefix="APP_")
logger.info(f"Loaded configuration with {len(settings.get_all())} settings")

# Environment-specific settings
ENV = os.environ.get("APP_ENV", "development").lower()
DEBUG = settings.get("debug_mode", ENV != "production")
API_PREFIX = settings.get("api.api_prefix", "/api/v1")

# Database settings
DB_TYPE = settings.get("database.type", "sqlite")
if DB_TYPE == "sqlite":
    DB_PATH = settings.get("database.sqlite_path", str(DATA_DIR / "investment.db"))
    DATABASE_URL = f"sqlite:///{DB_PATH}"
else:
    DATABASE_URL = settings.get("database.postgres_dsn", "")

# API keys
API_KEYS = {
    "newsapi": settings.get("api_keys.newsapi", os.environ.get("NEWSAPI_KEY", "")),
    "alpha_vantage": settings.get("api_keys.alpha_vantage", os.environ.get("ALPHA_VANTAGE_KEY", "")),
    "finnhub": settings.get("api_keys.finnhub", os.environ.get("FINNHUB_KEY", "")),
}

# Model settings
MODEL_SETTINGS = settings.get("models", {})

# Market data settings
MARKET_DATA_SETTINGS = settings.get("market_data", {})
DEFAULT_DATA_SOURCE = MARKET_DATA_SETTINGS.get("default_source", "yfinance")
DEFAULT_PERIOD = MARKET_DATA_SETTINGS.get("default_period", "1y")
DEFAULT_INTERVAL = MARKET_DATA_SETTINGS.get("default_interval", "1d")
CACHE_EXPIRY = MARKET_DATA_SETTINGS.get("cache_expiry", 24)  # hours

# Storage paths
STORAGE_SETTINGS = settings.get("storage", {})
DATA_DIR = Path(STORAGE_SETTINGS.get("data_dir", str(DATA_DIR)))
MODELS_DIR = Path(STORAGE_SETTINGS.get("models_dir", str(MODELS_DIR)))
CACHE_DIR = Path(STORAGE_SETTINGS.get("cache_dir", str(CACHE_DIR)))
RESULTS_DIR = Path(STORAGE_SETTINGS.get("results_dir", str(RESULTS_DIR)))

# Ensure directories exist after potentially updating paths from config
for directory in [DATA_DIR, MODELS_DIR, CACHE_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API settings
API_SETTINGS = settings.get("api", {})
API_HOST = API_SETTINGS.get("host", "0.0.0.0") 
API_PORT = int(API_SETTINGS.get("port", 8000))
API_WORKERS = int(API_SETTINGS.get("workers", 4))
API_TIMEOUT = int(API_SETTINGS.get("timeout", 60))
CORS_ORIGINS = API_SETTINGS.get("cors_origins", ["*"])

# Visualization settings
VISUALIZATION_SETTINGS = settings.get("visualization", {})
VIZ_THEME = VISUALIZATION_SETTINGS.get("theme", "dark")
VIZ_DEFAULT_FIGSIZE = VISUALIZATION_SETTINGS.get("default_figsize", [12, 8])
VIZ_DPI = VISUALIZATION_SETTINGS.get("dpi", 100)
VIZ_SAVE_FORMAT = VISUALIZATION_SETTINGS.get("save_format", ["png", "pdf"])

def get_api_key(service: str) -> str:
    """Get API key for a service."""
    return API_KEYS.get(service, "")

def get_model_setting(model_type: str, setting: str, default: Optional[any] = None) -> any:
    """Get setting for a specific model type."""
    model_config = MODEL_SETTINGS.get(model_type, {})
    return model_config.get(setting, default)

def get_full_path(relative_path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """Get full path relative to a base directory."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    
    if base_dir is None:
        base_dir = BASE_DIR
        
    return base_dir / path 