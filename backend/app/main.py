"""
Main FastAPI application entry point.

This module initializes the FastAPI application and includes all the routes.
"""

import logging
from fastapi import FastAPI, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
import time
import os
from pathlib import Path
import asyncio

# Import configuration
from app.config import (
    API_PREFIX, 
    CORS_ORIGINS, 
    ENV, 
    DEBUG, 
    RESULTS_DIR,
    DATA_DIR
)

# Import database utilities
from app.utils.database_utils import get_db, optimize_database_tables, create_indices
from sqlalchemy.ext.asyncio import AsyncSession

# Import archived data API
from app.api import archived_data

# Set up logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create the FastAPI application
app = FastAPI(
    title="Investment Recommendation System API",
    description="API for investment data analysis and recommendations",
    version="0.1.0",
    debug=DEBUG,
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc",
    openapi_url=f"{API_PREFIX}/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression middleware for response optimization
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers and log slow requests."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log requests that take more than 1 second as potential performance issues
    if process_time > 1.0:
        logger.warning(f"Slow request: {request.method} {request.url.path} took {process_time:.2f}s")
    
    return response

# Rate limiting middleware (simple in-memory implementation)
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware based on client IP."""
    client_ip = request.client.host if request.client else "unknown"
    
    # Implement a simple token bucket algorithm in a production system
    # For now, just track requests for demonstration purposes
    logger.debug(f"Request from {client_ip}: {request.method} {request.url.path}")
    
    # Continue with request processing
    return await call_next(request)

@app.on_event("startup")
async def startup_event():
    """Initialize resources and perform startup tasks."""
    logger.info(f"Starting Investment Recommendation System API in {ENV} mode")
    
    # Initialize database and create indices on startup
    try:
        # Get a database session
        db_gen = get_db()
        db = await anext(db_gen)
        
        # Create indices for optimized queries
        await create_indices(db)
        
        # Optimize database tables
        await optimize_database_tables(db)
        
        await db.close()
        logger.info("Database setup completed successfully")
    except Exception as e:
        logger.error(f"Error during database setup: {str(e)}")
    
    # Ensure necessary directories exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Load API routes
    from app.api.v1 import api_router
    app.include_router(api_router, prefix=API_PREFIX)
    
    # Mount static files for dashboard if available
    dashboard_path = Path(RESULTS_DIR) / "dashboard"
    if dashboard_path.exists():
        app.mount("/dashboard", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")
        logger.info(f"Dashboard mounted at /dashboard from {dashboard_path}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown."""
    logger.info("Shutting down Investment Recommendation System API")
    
    # Additional cleanup can be added here
    # For example, closing any background task pools

# Health check endpoint
@app.get(f"{API_PREFIX}/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "environment": ENV}

# Debug info endpoint (only available in development)
if ENV == "development":
    @app.get(f"{API_PREFIX}/debug/info")
    async def debug_info():
        """Debug endpoint with system information."""
        import platform
        import sys
        
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "environment": ENV,
            "debug_mode": DEBUG,
            "api_prefix": API_PREFIX,
        }

# Mount static files for results and data
if os.path.exists(RESULTS_DIR):
    app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

if os.path.exists(Path(DATA_DIR) / "public"):
    app.mount(
        "/data", 
        StaticFiles(directory=Path(DATA_DIR) / "public"), 
        name="public_data"
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# Include API routers
from app.api.v1.endpoints import market_data
app.include_router(market_data.router, prefix=f"{API_PREFIX}/market-data")
# app.include_router(sentiment.router, prefix=f"{API_PREFIX}/sentiment")
# app.include_router(forecast.router, prefix=f"{API_PREFIX}/forecast")
# app.include_router(portfolio.router, prefix=f"{API_PREFIX}/portfolio")

# Add archived data router
app.include_router(archived_data.router, prefix="/api/archive", tags=["archive"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint that redirects to documentation."""
    return {
        "message": "Investment Recommendation System API",
        "documentation": f"{API_PREFIX}/docs",
        "environment": ENV,
    } 