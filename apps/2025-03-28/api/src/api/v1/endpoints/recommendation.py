from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from src.api.deps import get_current_active_user, get_db
from src.models.user import User
from src.services.recommendation import recommendation_service
from src.crud.portfolio import portfolio_crud

router = APIRouter()

@router.post("/generate")
async def generate_recommendation(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    investment_amount: float,
    risk_tolerance: int = None,
    investment_horizon: int = None,
    target_return: float = None,
) -> Any:
    """
    Generate investment recommendations based on user preferences.
    """
    # Use user's stored preferences if not provided
    if risk_tolerance is None:
        risk_tolerance = current_user.risk_tolerance or 3
    if investment_horizon is None:
        investment_horizon = current_user.investment_horizon or 60  # 5 years default

    if not 1 <= risk_tolerance <= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Risk tolerance must be between 1 and 5",
        )

    if investment_amount <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Investment amount must be positive",
        )

    recommendation = await recommendation_service.generate_recommendation(
        risk_tolerance=risk_tolerance,
        investment_amount=investment_amount,
        investment_horizon=investment_horizon,
        target_return=target_return,
    )

    if not recommendation["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=recommendation["message"],
        )

    return recommendation

@router.post("/{portfolio_id}/rebalance")
async def rebalance_portfolio(
    *,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    portfolio_id: UUID,
) -> Any:
    """
    Generate rebalancing recommendations for an existing portfolio.
    """
    portfolio = await portfolio_crud.get_portfolio_with_holdings(
        db, portfolio_id=portfolio_id
    )
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found",
        )
    if portfolio.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )

    # Calculate current portfolio value and get symbols
    symbols = [holding.symbol for holding in portfolio.holdings]
    current_value = sum(
        holding.quantity * holding.last_price
        for holding in portfolio.holdings
        if holding.last_price is not None
    )

    if not symbols or current_value <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Portfolio has no valid holdings",
        )

    # Generate new recommendation with current portfolio value
    recommendation = await recommendation_service.generate_recommendation(
        risk_tolerance=current_user.risk_tolerance or 3,
        investment_amount=current_value,
        investment_horizon=current_user.investment_horizon or 60,
        target_return=portfolio.target_return,
    )

    if not recommendation["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=recommendation["message"],
        )

    # Calculate rebalancing trades
    current_holdings = {
        holding.symbol: {
            "shares": holding.quantity,
            "price": holding.last_price,
            "current_weight": holding.current_weight,
        }
        for holding in portfolio.holdings
    }

    trades = []
    for allocation in recommendation["portfolio"]["allocations"]:
        symbol = allocation["symbol"]
        target_shares = allocation["shares"]
        current = current_holdings.get(symbol, {"shares": 0})
        
        diff_shares = target_shares - current["shares"]
        if diff_shares != 0:
            trades.append({
                "symbol": symbol,
                "action": "buy" if diff_shares > 0 else "sell",
                "shares": abs(diff_shares),
                "price": allocation["price"],
                "amount": abs(diff_shares * allocation["price"]),
                "target_weight": allocation["weight"],
                "current_weight": current.get("current_weight", 0),
            })

    return {
        "current_portfolio": {
            "value": current_value,
            "holdings": current_holdings,
        },
        "target_portfolio": recommendation["portfolio"],
        "trades": trades,
    } 