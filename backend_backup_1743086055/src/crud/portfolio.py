from typing import List, Optional
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from src.crud.base import CRUDBase
from src.models.portfolio import (
    Portfolio,
    PortfolioCreate,
    PortfolioUpdate,
    Holding,
    HoldingCreate,
    HoldingUpdate,
)

class CRUDPortfolio(CRUDBase[Portfolio, PortfolioCreate, PortfolioUpdate]):
    async def get_user_portfolios(
        self, db: AsyncSession, *, user_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Portfolio]:
        statement = (
            select(Portfolio)
            .where(Portfolio.user_id == user_id)
            .offset(skip)
            .limit(limit)
        )
        results = await db.execute(statement)
        return results.scalars().all()

    async def get_portfolio_with_holdings(
        self, db: AsyncSession, *, portfolio_id: UUID
    ) -> Optional[Portfolio]:
        statement = select(Portfolio).where(Portfolio.id == portfolio_id)
        results = await db.execute(statement)
        return results.scalar_one_or_none()

class CRUDHolding(CRUDBase[Holding, HoldingCreate, HoldingUpdate]):
    async def get_portfolio_holdings(
        self, db: AsyncSession, *, portfolio_id: UUID, skip: int = 0, limit: int = 100
    ) -> List[Holding]:
        statement = (
            select(Holding)
            .where(Holding.portfolio_id == portfolio_id)
            .offset(skip)
            .limit(limit)
        )
        results = await db.execute(statement)
        return results.scalars().all()

    async def update_holding_weights(
        self, db: AsyncSession, *, portfolio_id: UUID, holdings: List[Holding]
    ) -> List[Holding]:
        total_value = sum(holding.total_value or 0 for holding in holdings)
        
        for holding in holdings:
            if total_value > 0 and holding.total_value:
                holding.current_weight = holding.total_value / total_value
            else:
                holding.current_weight = 0
            db.add(holding)
        
        await db.commit()
        for holding in holdings:
            await db.refresh(holding)
        
        return holdings

portfolio_crud = CRUDPortfolio(Portfolio)
holding_crud = CRUDHolding(Holding) 