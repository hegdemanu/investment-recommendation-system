"""Recommendation model module."""
from typing import TYPE_CHECKING
from sqlalchemy import Column, ForeignKey, String, Float, DateTime, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.db.base_class import Base

if TYPE_CHECKING:
    from .user import User  # noqa
    from .stock import Stock  # noqa

class RecommendationType(str, enum.Enum):
    """Recommendation type enum."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class Recommendation(Base):
    """Recommendation model."""
    
    recommendation_type = Column(Enum(RecommendationType), nullable=False)
    confidence_score = Column(Float, nullable=False)
    price_target = Column(Float)
    reasoning = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Foreign Keys
    user_id = Column(String, ForeignKey("user.id"), nullable=False)
    stock_id = Column(String, ForeignKey("stock.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="recommendations")
    stock = relationship("Stock", back_populates="recommendations")
