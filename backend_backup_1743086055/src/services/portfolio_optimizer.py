import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from scipy.optimize import minimize

from src.core.logger import logger
from src.services.market_data import market_data_service

class PortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate

    async def get_historical_returns(
        self,
        symbols: List[str],
        start_date: datetime = None,
        end_date: datetime = None,
        period: str = "1y",
    ) -> pd.DataFrame:
        """
        Get historical returns for a list of symbols.
        """
        try:
            dfs = []
            for symbol in symbols:
                df = await market_data_service.get_historical_data(
                    symbol, start_date, end_date, period
                )
                if not df.empty:
                    dfs.append(df["Close"].rename(symbol))
            
            if dfs:
                prices = pd.concat(dfs, axis=1)
                returns = prices.pct_change().dropna()
                return returns
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error calculating historical returns: {str(e)}")
            return pd.DataFrame()

    def calculate_portfolio_metrics(
        self, returns: pd.DataFrame, weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate portfolio metrics including return, volatility, and Sharpe ratio.
        """
        try:
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_std = np.sqrt(
                np.dot(weights.T, np.dot(returns.cov() * 252, weights))
            )
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
            
            return {
                "expected_return": float(portfolio_return),
                "volatility": float(portfolio_std),
                "sharpe_ratio": float(sharpe_ratio),
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {
                "expected_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
            }

    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        target_return: float = None,
        target_risk: float = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Optimize portfolio weights based on modern portfolio theory.
        """
        try:
            n_assets = returns.shape[1]
            bounds = tuple((0, 1) for _ in range(n_assets))
            constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

            if target_return is not None:
                constraints.append({
                    "type": "eq",
                    "fun": lambda x: np.sum(returns.mean() * x) * 252 - target_return,
                })

            if target_risk is not None:
                constraints.append({
                    "type": "eq",
                    "fun": lambda x: np.sqrt(
                        np.dot(x.T, np.dot(returns.cov() * 252, x))
                    ) - target_risk,
                })

            def objective(weights):
                portfolio_metrics = self.calculate_portfolio_metrics(returns, weights)
                return -portfolio_metrics["sharpe_ratio"]

            initial_weights = np.array([1.0 / n_assets] * n_assets)
            result = minimize(
                objective,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                optimal_weights = result.x
                metrics = self.calculate_portfolio_metrics(returns, optimal_weights)
                return optimal_weights, metrics
            else:
                logger.error(f"Portfolio optimization failed: {result.message}")
                return initial_weights, self.calculate_portfolio_metrics(returns, initial_weights)

        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            initial_weights = np.array([1.0 / n_assets] * n_assets)
            return initial_weights, self.calculate_portfolio_metrics(returns, initial_weights)

    async def get_efficient_frontier(
        self,
        symbols: List[str],
        n_points: int = 100,
        period: str = "1y",
    ) -> List[Dict[str, float]]:
        """
        Generate efficient frontier points.
        """
        try:
            returns = await self.get_historical_returns(symbols, period=period)
            if returns.empty:
                return []

            min_return = returns.mean().min() * 252
            max_return = returns.mean().max() * 252
            target_returns = np.linspace(min_return, max_return, n_points)

            frontier_points = []
            for target_return in target_returns:
                weights, metrics = self.optimize_portfolio(returns, target_return=target_return)
                frontier_points.append({
                    "return": metrics["expected_return"],
                    "risk": metrics["volatility"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "weights": {symbol: float(weight) for symbol, weight in zip(symbols, weights)},
                })

            return frontier_points

        except Exception as e:
            logger.error(f"Error generating efficient frontier: {str(e)}")
            return []

portfolio_optimizer = PortfolioOptimizer() 