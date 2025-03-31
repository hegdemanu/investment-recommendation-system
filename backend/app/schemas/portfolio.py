from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel


# Investment schemas
class InvestmentBase(BaseModel):
    symbol: str
    name: str
    quantity: float
    purchase_price: float
    purchase_date: datetime


class InvestmentCreate(InvestmentBase):
    pass


class InvestmentUpdate(BaseModel):
    symbol: Optional[str] = None
    name: Optional[str] = None
    quantity: Optional[float] = None
    purchase_price: Optional[float] = None
    purchase_date: Optional[datetime] = None
    current_price: Optional[float] = None


class InvestmentInDBBase(InvestmentBase):
    id: int
    portfolio_id: int
    current_price: Optional[float] = None
    last_updated: Optional[datetime] = None

    class Config:
        from_attributes = True


class Investment(InvestmentInDBBase):
    pass


# Portfolio schemas
class PortfolioBase(BaseModel):
    name: str
    description: Optional[str] = None


class PortfolioCreate(PortfolioBase):
    pass


class PortfolioUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class PortfolioInDBBase(PortfolioBase):
    id: int
    owner_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class Portfolio(PortfolioInDBBase):
    investments: List[Investment] = [] 