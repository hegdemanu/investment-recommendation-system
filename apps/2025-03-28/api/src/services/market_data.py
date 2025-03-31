import yfinance as yf
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import httpx

from src.core.config import settings
from src.core.logger import logger

class MarketDataService:
    def __init__(self):
        self.finnhub_client = httpx.AsyncClient(
            base_url="https://finnhub.io/api/v1",
            params={"token": settings.FINNHUB_API_KEY},
        )

    async def get_stock_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote for a stock.
        """
        try:
            response = await self.finnhub_client.get(f"/quote", params={"symbol": symbol})
            response.raise_for_status()
            data = response.json()
            return {
                "current_price": data["c"],
                "change": data["d"],
                "percent_change": data["dp"],
                "high": data["h"],
                "low": data["l"],
                "open": data["o"],
                "previous_close": data["pc"],
                "timestamp": data["t"],
            }
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            return {}

    async def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get company information for a stock.
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return {
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "beta": info.get("beta", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "description": info.get("longBusinessSummary", ""),
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}

    async def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "1y",
    ) -> pd.DataFrame:
        """
        Get historical price data for a stock.
        """
        try:
            stock = yf.Ticker(symbol)
            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

    async def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get real-time quotes for multiple stocks.
        """
        tasks = [self.get_stock_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))

    async def get_market_news(self, category: str = "general") -> List[Dict[str, Any]]:
        """
        Get market news.
        """
        try:
            response = await self.finnhub_client.get(
                f"/news", params={"category": category}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching market news: {str(e)}")
            return []

    async def close(self):
        """
        Close the HTTP client.
        """
        await self.finnhub_client.aclose()

market_data_service = MarketDataService() 