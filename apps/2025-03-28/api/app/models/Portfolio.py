"""Portfolio model module."""
from typing import TYPE_CHECKING
from sqlalchemy import Column, ForeignKey, String, Float, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base

if TYPE_CHECKING:
    from .user import User  # noqa
    from .stock import Stock  # noqa

class Portfolio(Base):
    """Portfolio model."""
    
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    total_value = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Foreign Keys
    user_id = Column(String, ForeignKey("user.id"), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    stocks = relationship("Stock", back_populates="portfolio")
