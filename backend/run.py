#!/usr/bin/env python3
"""
Script to run the FastAPI application.

This module starts the FastAPI server using uvicorn.
"""

import uvicorn
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import configuration
from app.config import API_HOST, API_PORT, API_WORKERS, API_TIMEOUT, DEBUG, ENV

# Set up logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Run the FastAPI application."""
    logger.info(f"Starting server in {ENV} environment")
    
    # Configure uvicorn
    uvicorn_config = {
        "host": API_HOST,
        "port": API_PORT,
        "workers": API_WORKERS if ENV == "production" else 1,
        "reload": ENV != "production",
        "log_level": "debug" if DEBUG else "info",
        "timeout_keep_alive": API_TIMEOUT,
    }
    
    logger.info(f"Server configuration: {uvicorn_config}")
    
    # Start server
    uvicorn.run("app.main:app", **uvicorn_config)
    
if __name__ == "__main__":
    main() 