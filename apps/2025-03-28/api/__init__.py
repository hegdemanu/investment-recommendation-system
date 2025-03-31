"""
Investment Recommendation System

This package contains the main application code for the Investment Recommendation System.

The system has been refactored into a more modular and maintainable structure:

- app/core/ - Core business logic for stock analysis and predictions
- app/dashboard/ - Dashboard generation and visualization
- app/api/ - API endpoints for the web interface
- app/utils/ - Utility functions for file operations, data processing, etc.
- app/templates/ - HTML templates
- app/static/ - Static assets (CSS, JS, images)

The config/ directory contains centralized configuration settings.
"""

__version__ = '1.1.0'

# Initialize directories
from config.settings import ensure_directories
ensure_directories()

def create_app():
    """Create and configure the FastAPI app"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    # Create FastAPI app
    app = FastAPI(
        title="Investment Recommendation System",
        description="API for stock price predictions and sentiment analysis",
        version=__version__
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Import and include routers
    from app.api.routes import router as api_router
    app.include_router(api_router, prefix="/api/v1", tags=["api"])
    
    return app 