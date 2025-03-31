"""
Market data API endpoints.

This module provides API endpoints for fetching and analyzing market data.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import json

# Import services
from app.services.market_data import MarketDataService

# Import configuration
from app.config import API_PREFIX, DEFAULT_PERIOD, DEFAULT_INTERVAL

# Setup router
router = APIRouter(tags=["market_data"])

# Set up logging
logger = logging.getLogger(__name__)

# Helper function to get market data service
async def get_market_data_service():
    """Dependency for getting the market data service."""
    service = MarketDataService()
    return service

@router.get("/stocks/{ticker}")
async def get_stock_data(
    ticker: str,
    period: Optional[str] = Query(DEFAULT_PERIOD, description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    interval: Optional[str] = Query(DEFAULT_INTERVAL, description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)"),
    include_indicators: bool = Query(True, description="Include technical indicators"),
    force_refresh: bool = Query(False, description="Force refresh data from source"),
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """
    Get stock data for a ticker.
    
    - **ticker**: Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
    - **period**: Time period to fetch
    - **interval**: Data interval
    - **include_indicators**: Whether to include technical indicators
    - **force_refresh**: Whether to force refresh data from source
    """
    try:
        data = await market_data_service.get_stock_data(
            ticker=ticker,
            period=period,
            interval=interval,
            include_indicators=include_indicators,
            force_refresh=force_refresh
        )
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker: {ticker}")
        
        # Convert to JSON-serializable format
        result = {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "data_points": len(data),
            "start_date": data.index.min().strftime("%Y-%m-%d"),
            "end_date": data.index.max().strftime("%Y-%m-%d"),
            "columns": list(data.columns),
            "data": json.loads(data.reset_index().to_json(orient="records", date_format="iso"))
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

@router.get("/stocks/{ticker}/analysis")
async def analyze_stock(
    ticker: str,
    force_refresh: bool = Query(False, description="Force refresh data from source"),
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """
    Perform comprehensive analysis on a stock.
    
    - **ticker**: Stock ticker symbol
    - **force_refresh**: Whether to force refresh data from source
    """
    try:
        analysis = await market_data_service.analyze_stock(
            ticker=ticker,
            force_refresh=force_refresh
        )
        
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        
        return analysis
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error analyzing stock {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing stock: {str(e)}")

@router.get("/stocks/{ticker}/beta")
async def get_beta(
    ticker: str,
    market_index: str = Query("^GSPC", description="Market index symbol"),
    period: int = Query(252, description="Number of days to use for calculation"),
    force_refresh: bool = Query(False, description="Force refresh data from source"),
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """
    Calculate beta for a stock.
    
    - **ticker**: Stock ticker symbol
    - **market_index**: Market index symbol
    - **period**: Number of days to use for calculation
    - **force_refresh**: Whether to force refresh data from source
    """
    try:
        beta = await market_data_service.get_stock_beta(
            ticker=ticker,
            market_index=market_index,
            period=period,
            force_refresh=force_refresh
        )
        
        return {
            "ticker": ticker,
            "market_index": market_index,
            "period": period,
            "beta": beta
        }
    
    except Exception as e:
        logger.error(f"Error calculating beta for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating beta: {str(e)}")

@router.get("/stocks")
async def get_multiple_stocks(
    tickers: str = Query(..., description="Comma-separated list of ticker symbols"),
    period: Optional[str] = Query(DEFAULT_PERIOD, description="Time period"),
    interval: Optional[str] = Query(DEFAULT_INTERVAL, description="Data interval"),
    include_indicators: bool = Query(False, description="Include technical indicators"),
    force_refresh: bool = Query(False, description="Force refresh data from source"),
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """
    Get data for multiple stocks.
    
    - **tickers**: Comma-separated list of ticker symbols
    - **period**: Time period to fetch
    - **interval**: Data interval
    - **include_indicators**: Whether to include technical indicators
    - **force_refresh**: Whether to force refresh data from source
    """
    try:
        # Parse tickers
        ticker_list = [t.strip() for t in tickers.split(",")]
        
        if not ticker_list:
            raise HTTPException(status_code=400, detail="No tickers provided")
        
        data = await market_data_service.get_multiple_stocks(
            tickers=ticker_list,
            period=period,
            interval=interval,
            include_indicators=include_indicators,
            force_refresh=force_refresh
        )
        
        if not data:
            raise HTTPException(status_code=404, detail="No data found for any of the provided tickers")
        
        # Convert to JSON-serializable format
        result = {
            "tickers": ticker_list,
            "period": period,
            "interval": interval,
            "found_tickers": list(data.keys()),
            "missing_tickers": [t for t in ticker_list if t not in data]
        }
        
        # Add data summaries
        stock_data = {}
        for ticker, df in data.items():
            stock_data[ticker] = {
                "data_points": len(df),
                "start_date": df.index.min().strftime("%Y-%m-%d"),
                "end_date": df.index.max().strftime("%Y-%m-%d"),
                "latest_price": float(df["Close"].iloc[-1]),
                "data": json.loads(df.reset_index().to_json(orient="records", date_format="iso"))
            }
        
        result["stocks"] = stock_data
        
        return result
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching multiple stocks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching multiple stocks: {str(e)}")

@router.get("/market-index/{index_symbol}")
async def get_market_index(
    index_symbol: str,
    period: Optional[str] = Query(DEFAULT_PERIOD, description="Time period"),
    include_indicators: bool = Query(True, description="Include technical indicators"),
    force_refresh: bool = Query(False, description="Force refresh data from source"),
    market_data_service: MarketDataService = Depends(get_market_data_service)
):
    """
    Get market index data.
    
    - **index_symbol**: Index symbol (e.g., ^GSPC, ^DJI, ^IXIC) or name (sp500, dow, nasdaq)
    - **period**: Time period to fetch
    - **include_indicators**: Whether to include technical indicators
    - **force_refresh**: Whether to force refresh data from source
    """
    try:
        data = await market_data_service.get_market_index(
            index_symbol=index_symbol,
            period=period,
            include_indicators=include_indicators,
            force_refresh=force_refresh
        )
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for market index: {index_symbol}")
        
        # Convert to JSON-serializable format
        result = {
            "index": index_symbol,
            "period": period,
            "data_points": len(data),
            "start_date": data.index.min().strftime("%Y-%m-%d"),
            "end_date": data.index.max().strftime("%Y-%m-%d"),
            "latest_value": float(data["Close"].iloc[-1]),
            "change_percent": float(((data["Close"].iloc[-1] / data["Close"].iloc[0]) - 1) * 100),
            "data": json.loads(data.reset_index().to_json(orient="records", date_format="iso"))
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error fetching market index data for {index_symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching market index data: {str(e)}") 