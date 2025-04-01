from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship
from uuid import UUID
from src.models.base import BaseModel

class PortfolioBase(SQLModel):
    name: str = Field(index=True)
    description: Optional[str] = None
    target_risk: Optional[float] = None
    target_return: Optional[float] = None
    investment_amount: float
    currency: str = Field(default="USD")
    rebalancing_frequency: str = Field(default="monthly")  # monthly, quarterly, annually

class Portfolio(BaseModel, PortfolioBase, table=True):
    user_id: UUID = Field(foreign_key="user.id")
    user: "User" = Relationship(back_populates="portfolios")
    holdings: List["Holding"] = Relationship(back_populates="portfolio")

class PortfolioCreate(PortfolioBase):
    pass

class PortfolioUpdate(SQLModel):
    name: Optional[str] = None
    description: Optional[str] = None
    target_risk: Optional[float] = None
    target_return: Optional[float] = None
    investment_amount: Optional[float] = None
    currency: Optional[str] = None
    rebalancing_frequency: Optional[str] = None

class PortfolioResponse(PortfolioBase):
    id: UUID
    user_id: UUID
    holdings: List["HoldingResponse"] = []

    class Config:
        orm_mode = True

class HoldingBase(SQLModel):
    symbol: str = Field(index=True)
    quantity: float
    target_weight: float
    current_weight: Optional[float] = None
    average_price: Optional[float] = None
    last_price: Optional[float] = None
    total_value: Optional[float] = None

class Holding(BaseModel, HoldingBase, table=True):
    portfolio_id: UUID = Field(foreign_key="portfolio.id")
    portfolio: Portfolio = Relationship(back_populates="holdings")

class HoldingCreate(HoldingBase):
    pass

class HoldingUpdate(SQLModel):
    quantity: Optional[float] = None
    target_weight: Optional[float] = None
    current_weight: Optional[float] = None
    average_price: Optional[float] = None
    last_price: Optional[float] = None
    total_value: Optional[float] = None

class HoldingResponse(HoldingBase):
    id: UUID
    portfolio_id: UUID

    class Config:
        orm_mode = True

PortfolioResponse.update_forward_refs(HoldingResponse=HoldingResponse) 