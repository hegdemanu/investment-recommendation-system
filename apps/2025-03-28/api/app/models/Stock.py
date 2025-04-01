"""Stock model module."""
from typing import TYPE_CHECKING
from sqlalchemy import Column, ForeignKey, String, Float, Integer, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base

if TYPE_CHECKING:
    from .portfolio import Portfolio  # noqa
    from .recommendation import Recommendation  # noqa

class Stock(Base):
    """Stock model."""
    
    symbol = Column(String, index=True, nullable=False)
    company_name = Column(String)
    quantity = Column(Integer, default=0)
    purchase_price = Column(Float)
    current_price = Column(Float)
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Foreign Keys
    portfolio_id = Column(String, ForeignKey("portfolio.id"), nullable=False)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="stocks")
    recommendations = relationship("Recommendation", back_populates="stock")
