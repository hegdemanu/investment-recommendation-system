"""
API router initialization for API v1.

This module initializes the router for API v1 endpoints.
"""

from fastapi import APIRouter

# Create API v1 router
api_router = APIRouter()

# Import and include other routers
from .stocks import router as stocks_router
from .predict import router as predict_router
from .sentiment import router as sentiment_router
from .trading import router as trading_router

# Include routers
api_router.include_router(stocks_router, prefix="/stocks", tags=["stocks"])
api_router.include_router(predict_router, prefix="/predict", tags=["predict"])
api_router.include_router(sentiment_router, prefix="/sentiment", tags=["sentiment"])
api_router.include_router(trading_router, prefix="/trading", tags=["trading"]) 