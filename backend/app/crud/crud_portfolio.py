from typing import List, Optional

from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.portfolio import Portfolio, Investment
from app.schemas.portfolio import PortfolioCreate, PortfolioUpdate, InvestmentCreate, InvestmentUpdate


class CRUDPortfolio(CRUDBase[Portfolio, PortfolioCreate, PortfolioUpdate]):
    def create_with_owner(
        self, db: Session, *, obj_in: PortfolioCreate, owner_id: int
    ) -> Portfolio:
        obj_in_data = obj_in.model_dump()
        db_obj = self.model(**obj_in_data, owner_id=owner_id)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_multi_by_owner(
        self, db: Session, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[Portfolio]:
        return (
            db.query(self.model)
            .filter(Portfolio.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
            .all()
        )


class CRUDInvestment(CRUDBase[Investment, InvestmentCreate, InvestmentUpdate]):
    def create_with_portfolio(
        self, db: Session, *, obj_in: InvestmentCreate, portfolio_id: int
    ) -> Investment:
        obj_in_data = obj_in.model_dump()
        db_obj = self.model(**obj_in_data, portfolio_id=portfolio_id)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_multi_by_portfolio(
        self, db: Session, *, portfolio_id: int, skip: int = 0, limit: int = 100
    ) -> List[Investment]:
        return (
            db.query(self.model)
            .filter(Investment.portfolio_id == portfolio_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
        
    def get_by_symbol(
        self, db: Session, *, portfolio_id: int, symbol: str
    ) -> Optional[Investment]:
        return (
            db.query(self.model)
            .filter(Investment.portfolio_id == portfolio_id, Investment.symbol == symbol)
            .first()
        )


portfolio = CRUDPortfolio(Portfolio)
investment = CRUDInvestment(Investment) 