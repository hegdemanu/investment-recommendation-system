import pytest
import os
from config.settings import (
    API_CONFIG,
    DB_CONFIG,
    CACHE_CONFIG,
    MODEL_CONFIG,
    LOGGING_CONFIG,
    ensure_directories
)

def test_api_config():
    assert isinstance(API_CONFIG, dict)
    assert "host" in API_CONFIG
    assert "port" in API_CONFIG
    assert "debug" in API_CONFIG
    assert isinstance(API_CONFIG["port"], int)
    assert isinstance(API_CONFIG["debug"], bool)

def test_db_config():
    assert isinstance(DB_CONFIG, dict)
    assert "host" in DB_CONFIG
    assert "port" in DB_CONFIG
    assert "name" in DB_CONFIG
    assert "user" in DB_CONFIG
    assert "password" in DB_CONFIG
    assert isinstance(DB_CONFIG["port"], int)

def test_cache_config():
    assert isinstance(CACHE_CONFIG, dict)
    assert "type" in CACHE_CONFIG
    assert "url" in CACHE_CONFIG
    assert "ttl" in CACHE_CONFIG
    assert isinstance(CACHE_CONFIG["ttl"], int)

def test_model_config():
    assert isinstance(MODEL_CONFIG, dict)
    assert "cache_dir" in MODEL_CONFIG
    assert "training_data_dir" in MODEL_CONFIG
    assert "prediction_data_dir" in MODEL_CONFIG
    assert all(isinstance(path, str) for path in MODEL_CONFIG.values())

def test_logging_config():
    assert isinstance(LOGGING_CONFIG, dict)
    assert "level" in LOGGING_CONFIG
    assert "file" in LOGGING_CONFIG
    assert isinstance(LOGGING_CONFIG["level"], str)

def test_ensure_directories():
    # Test directory creation
    ensure_directories()
    
    # Check if directories exist
    assert os.path.exists("data")
    assert os.path.exists("models")
    assert os.path.exists("logs")
    assert os.path.exists("results")
    
    # Check if subdirectories exist
    assert os.path.exists(os.path.join("data", "training"))
    assert os.path.exists(os.path.join("data", "predictions"))
    assert os.path.exists(os.path.join("models", "cache"))
    assert os.path.exists(os.path.join("results", "dashboard"))

def test_environment_variables():
    # Test required environment variables
    required_vars = [
        "ALPHA_VANTAGE_API_KEY",
        "NEWS_API_KEY",
        "FINNHUB_API_KEY"
    ]
    
    for var in required_vars:
        assert var in os.environ, f"Missing required environment variable: {var}"

def test_config_validation():
    # Test API port range
    assert 1024 <= API_CONFIG["port"] <= 65535
    
    # Test cache TTL
    assert CACHE_CONFIG["ttl"] > 0
    
    # Test logging level
    valid_logging_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    assert LOGGING_CONFIG["level"] in valid_logging_levels

def test_path_configurations():
    # Test if paths are absolute or relative to project root
    assert os.path.isabs(MODEL_CONFIG["cache_dir"]) or MODEL_CONFIG["cache_dir"].startswith("./")
    assert os.path.isabs(LOGGING_CONFIG["file"]) or LOGGING_CONFIG["file"].startswith("./") 