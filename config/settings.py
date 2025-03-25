"""
Configuration settings for the Investment Recommendation System.
This file centralizes all paths and configuration parameters.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Data directories
STOCKS_DATA_DIR = os.path.join(DATA_DIR, "stocks")
MUTUAL_FUNDS_DATA_DIR = os.path.join(DATA_DIR, "mutual_funds")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Results directories
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
TRAINING_DIR = os.path.join(RESULTS_DIR, "training")
VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
DASHBOARD_DIR = os.path.join(RESULTS_DIR, "dashboard")
DASHBOARD_JSON_DIR = os.path.join(DASHBOARD_DIR, "json")

# Dashboard settings
DASHBOARD_FILE = os.path.join(DASHBOARD_DIR, "investment_dashboard.html")

# Report directories
REPORTS_DIRS = {
    "training": TRAINING_DIR,
    "validation": VALIDATION_DIR,
    "reports": REPORTS_DIR,
    "models": MODELS_DIR
}

# Investment report settings
INVESTMENT_REPORT_PATH = os.path.join(REPORTS_DIR, "investment_report.html")

# Model settings
DEFAULT_HORIZON = 5  # Default prediction horizon in days
HORIZONS = [1, 3, 5, 7, 14, 21, 30]  # Available prediction horizons
SEQUENCE_LENGTH = 60  # Default sequence length for LSTM models

# Function to ensure all directories exist
def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR,
        STOCKS_DATA_DIR, MUTUAL_FUNDS_DATA_DIR, UPLOADS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR,
        REPORTS_DIR, TRAINING_DIR, VALIDATION_DIR, PREDICTIONS_DIR, DASHBOARD_DIR, DASHBOARD_JSON_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return True 