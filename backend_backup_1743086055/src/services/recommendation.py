from typing import List, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.core.logger import logger
from src.services.market_data import market_data_service
from src.services.portfolio_optimizer import portfolio_optimizer

class RecommendationService:
    def __init__(self):
        self.risk_profiles = {
            1: {"max_volatility": 0.10, "min_return": 0.04},  # Conservative
            2: {"max_volatility": 0.15, "min_return": 0.06},  # Moderate Conservative
            3: {"max_volatility": 0.20, "min_return": 0.08},  # Moderate
            4: {"max_volatility": 0.25, "min_return": 0.10},  # Moderate Aggressive
            5: {"max_volatility": 0.30, "min_return": 0.12},  # Aggressive
        }

    async def get_stock_universe(self, risk_tolerance: int) -> List[str]:
        """
        Get a list of stocks suitable for the given risk tolerance.
        """
        # This is a simplified example. In practice, you would have a more sophisticated
        # stock selection process based on various factors.
        base_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "BRK-B",  # Large Cap Tech & Finance
            "JNJ", "PG", "KO", "PEP", "WMT",           # Consumer Staples
            "VTI", "AGG", "VEA", "VWO", "BND",         # ETFs
            "GLD", "SLV", "DBC", "VNQ", "TIP",         # Alternative Assets
        ]

        if risk_tolerance <= 2:  # Conservative
            return base_stocks[5:15]  # Focus on stable stocks and bonds
        elif risk_tolerance <= 4:  # Moderate
            return base_stocks[0:15]  # Mix of growth and stable stocks
        else:  # Aggressive
            return base_stocks[0:10] + base_stocks[15:]  # Growth stocks and alternatives

    def filter_recommendations(
        self,
        frontier_points: List[Dict[str, float]],
        risk_tolerance: int,
        target_return: float = None,
    ) -> Dict[str, Any]:
        """
        Filter and select the most appropriate portfolio from the efficient frontier.
        """
        try:
            if not frontier_points:
                return {}

            risk_profile = self.risk_profiles[risk_tolerance]
            filtered_points = [
                point for point in frontier_points
                if point["risk"] <= risk_profile["max_volatility"]
                and point["return"] >= risk_profile["min_return"]
            ]

            if not filtered_points:
                return frontier_points[0]  # Return the minimum risk portfolio

            if target_return:
                # Find portfolio closest to target return
                return min(
                    filtered_points,
                    key=lambda x: abs(x["return"] - target_return)
                )
            else:
                # Return portfolio with highest Sharpe ratio
                return max(filtered_points, key=lambda x: x["sharpe_ratio"])

        except Exception as e:
            logger.error(f"Error filtering recommendations: {str(e)}")
            return {}

    async def generate_recommendation(
        self,
        risk_tolerance: int,
        investment_amount: float,
        investment_horizon: int,
        target_return: float = None,
    ) -> Dict[str, Any]:
        """
        Generate investment recommendations based on user preferences.
        """
        try:
            # Get suitable stock universe
            symbols = await self.get_stock_universe(risk_tolerance)

            # Generate efficient frontier
            frontier_points = await portfolio_optimizer.get_efficient_frontier(
                symbols, n_points=100, period="1y"
            )

            # Select optimal portfolio
            optimal_portfolio = self.filter_recommendations(
                frontier_points, risk_tolerance, target_return
            )

            if not optimal_portfolio:
                return {
                    "success": False,
                    "message": "Could not generate recommendations",
                }

            # Get current prices for allocation
            quotes = await market_data_service.get_multiple_quotes(symbols)
            
            # Calculate number of shares for each position
            allocations = []
            remaining_amount = investment_amount
            
            for symbol, weight in optimal_portfolio["weights"].items():
                if weight > 0.01:  # Filter out very small allocations
                    amount = investment_amount * weight
                    price = quotes[symbol].get("current_price", 0)
                    if price > 0:
                        shares = int(amount / price)
                        if shares > 0:
                            actual_amount = shares * price
                            remaining_amount -= actual_amount
                            allocations.append({
                                "symbol": symbol,
                                "shares": shares,
                                "amount": actual_amount,
                                "weight": weight,
                                "price": price,
                            })

            # Add market data for recommended stocks
            stock_info = {}
            for symbol in symbols:
                info = await market_data_service.get_stock_info(symbol)
                if info:
                    stock_info[symbol] = info

            return {
                "success": True,
                "portfolio": {
                    "expected_return": optimal_portfolio["return"],
                    "risk": optimal_portfolio["risk"],
                    "sharpe_ratio": optimal_portfolio["sharpe_ratio"],
                    "investment_amount": investment_amount,
                    "remaining_amount": remaining_amount,
                    "allocations": allocations,
                },
                "market_data": {
                    "quotes": quotes,
                    "stock_info": stock_info,
                },
                "risk_profile": {
                    "risk_tolerance": risk_tolerance,
                    "investment_horizon": investment_horizon,
                    "target_return": target_return,
                },
            }

        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                "success": False,
                "message": f"Error generating recommendations: {str(e)}",
            }

recommendation_service = RecommendationService() 