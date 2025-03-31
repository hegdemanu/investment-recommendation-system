"""
Stocks API endpoints.

This module provides API endpoints for stock data retrieval and analysis.
"""

from fastapi import APIRouter, Depends, Query, HTTPException, status
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.services.data_fetcher import get_stock_data, get_stock_info
from app.utils.database_utils import cache_query, get_db
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/{symbol}")
@cache_query(ttl_seconds=300)  # Cache results for 5 minutes
async def get_stock(
    symbol: str,
    period: str = Query("1mo", description="Data period (1d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    interval: str = Query("1d", description="Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)"),
    indicators: bool = Query(False, description="Include technical indicators"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get historical stock data for a given symbol.
    """
    try:
        logger.info(f"Retrieving stock data for {symbol}, period={period}, interval={interval}")
        data = await get_stock_data(symbol, period, interval, indicators, db)
        return {
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "indicators": indicators,
            "data": data
        }
    except Exception as e:
        logger.error(f"Error retrieving stock data for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stock data: {str(e)}"
        )

@router.get("/info/{symbol}")
@cache_query(ttl_seconds=3600)  # Cache results for 1 hour
async def get_stock_information(
    symbol: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get basic information about a stock.
    """
    try:
        logger.info(f"Retrieving stock info for {symbol}")
        info = await get_stock_info(symbol, db)
        return info
    except Exception as e:
        logger.error(f"Error retrieving stock info for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stock info: {str(e)}"
        )

@router.get("/search")
async def search_stocks(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results"),
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Search for stocks by name or symbol.
    """
    try:
        logger.info(f"Searching stocks with query: {query}")
        # Implementation depends on your stock data source
        # This is a placeholder
        results = []
        return results[:limit]
    except Exception as e:
        logger.error(f"Error searching stocks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching stocks: {str(e)}"
        ) 