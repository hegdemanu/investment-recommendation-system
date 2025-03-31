"""
Dashboard settings configuration
"""

# Dashboard configuration
DASHBOARD_CONFIG = {
    "theme": "light",
    "refresh_interval": 60,  # seconds
    "default_timeframe": "1d",
    "max_stocks_displayed": 10,
    "enable_notifications": True,
    "cache_timeout": 3600,  # seconds
}

# Chart settings
CHART_CONFIG = {
    "default_height": 400,
    "default_width": 800,
    "colors": {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "positive": "#2ca02c",
        "negative": "#d62728",
        "neutral": "#7f7f7f"
    },
}

# Data sources
DATA_SOURCES = {
    "market_data": "yfinance",
    "news": "newsapi",
    "sentiment": "finbert",
}

# Output directories
OUTPUT_DIR = "dashboard/output"
TEMPLATES_DIR = "dashboard/templates"
STATIC_DIR = "dashboard/static"

# Default stocks to display
DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"] 