from typing import Any, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_current_active_user, get_db
from src.crud.portfolio import portfolio_crud, holding_crud
from src.models.user import User
from src.models.portfolio import (
    Portfolio,
    PortfolioCreate,
    PortfolioUpdate,
    PortfolioResponse,
    Holding,
    HoldingCreate,
    HoldingUpdate,
    HoldingResponse,
)

router = APIRouter()

@router.get("/", response_model=List[PortfolioResponse])
async def get_portfolios(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve user's portfolios.
    """
    portfolios = await portfolio_crud.get_user_portfolios(
        db, user_id=current_user.id, skip=skip, limit=limit
    )
    return portfolios

@router.post("/", response_model=PortfolioResponse)
async def create_portfolio(
    *,
    db: AsyncSession = Depends(get_db),
    portfolio_in: PortfolioCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Create new portfolio.
    """
    portfolio = await portfolio_crud.create(
        db, obj_in=portfolio_in
    )
    portfolio.user_id = current_user.id
    db.add(portfolio)
    await db.commit()
    await db.refresh(portfolio)
    return portfolio

@router.get("/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(
    *,
    db: AsyncSession = Depends(get_db),
    portfolio_id: UUID,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get portfolio by ID.
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
    return portfolio

@router.put("/{portfolio_id}", response_model=PortfolioResponse)
async def update_portfolio(
    *,
    db: AsyncSession = Depends(get_db),
    portfolio_id: UUID,
    portfolio_in: PortfolioUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Update portfolio.
    """
    portfolio = await portfolio_crud.get(db, id=portfolio_id)
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
    portfolio = await portfolio_crud.update(
        db, db_obj=portfolio, obj_in=portfolio_in
    )
    return portfolio

@router.delete("/{portfolio_id}", response_model=PortfolioResponse)
async def delete_portfolio(
    *,
    db: AsyncSession = Depends(get_db),
    portfolio_id: UUID,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Delete portfolio.
    """
    portfolio = await portfolio_crud.get(db, id=portfolio_id)
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
    portfolio = await portfolio_crud.remove(db, id=portfolio_id)
    return portfolio

# Holdings endpoints

@router.get("/{portfolio_id}/holdings", response_model=List[HoldingResponse])
async def get_holdings(
    *,
    db: AsyncSession = Depends(get_db),
    portfolio_id: UUID,
    current_user: User = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve portfolio holdings.
    """
    portfolio = await portfolio_crud.get(db, id=portfolio_id)
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
    holdings = await holding_crud.get_portfolio_holdings(
        db, portfolio_id=portfolio_id, skip=skip, limit=limit
    )
    return holdings

@router.post("/{portfolio_id}/holdings", response_model=HoldingResponse)
async def create_holding(
    *,
    db: AsyncSession = Depends(get_db),
    portfolio_id: UUID,
    holding_in: HoldingCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Create new holding in portfolio.
    """
    portfolio = await portfolio_crud.get(db, id=portfolio_id)
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
    holding = await holding_crud.create(db, obj_in=holding_in)
    holding.portfolio_id = portfolio_id
    db.add(holding)
    await db.commit()
    await db.refresh(holding)
    return holding

@router.put("/{portfolio_id}/holdings/{holding_id}", response_model=HoldingResponse)
async def update_holding(
    *,
    db: AsyncSession = Depends(get_db),
    portfolio_id: UUID,
    holding_id: UUID,
    holding_in: HoldingUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Update holding in portfolio.
    """
    portfolio = await portfolio_crud.get(db, id=portfolio_id)
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
    holding = await holding_crud.get(db, id=holding_id)
    if not holding:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Holding not found",
        )
    if holding.portfolio_id != portfolio_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Holding does not belong to this portfolio",
        )
    holding = await holding_crud.update(
        db, db_obj=holding, obj_in=holding_in
    )
    return holding

@router.delete("/{portfolio_id}/holdings/{holding_id}", response_model=HoldingResponse)
async def delete_holding(
    *,
    db: AsyncSession = Depends(get_db),
    portfolio_id: UUID,
    holding_id: UUID,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Delete holding from portfolio.
    """
    portfolio = await portfolio_crud.get(db, id=portfolio_id)
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
    holding = await holding_crud.get(db, id=holding_id)
    if not holding:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Holding not found",
        )
    if holding.portfolio_id != portfolio_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Holding does not belong to this portfolio",
        )
    holding = await holding_crud.remove(db, id=holding_id)
    return holding