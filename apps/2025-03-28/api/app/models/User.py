"""User model module."""
from typing import TYPE_CHECKING
from sqlalchemy import Boolean, Column, String
from sqlalchemy.orm import relationship

from app.db.base_class import Base

if TYPE_CHECKING:
    from .portfolio import Portfolio  # noqa
    from .recommendation import Recommendation  # noqa

class User(Base):
    """User model."""
    
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, index=True)
    is_active = Column(Boolean(), default=True)
    is_superuser = Column(Boolean(), default=False)

    # Relationships
    portfolios = relationship("Portfolio", back_populates="user")
    recommendations = relationship("Recommendation", back_populates="user")
