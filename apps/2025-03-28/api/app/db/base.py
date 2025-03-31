"""SQLAlchemy base model module."""
from typing import Any
from datetime import datetime
from sqlalchemy.ext.declarative import as_declarative, declared_attr
from sqlalchemy import Column, Integer, DateTime

@as_declarative()
class Base:
    """Base class for all database models."""
    
    id: Any
    created_at: datetime
    updated_at: datetime
    
    # Generate __tablename__ automatically
    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name automatically."""
        return cls.__name__.lower() + 's'
    
    # Common columns for all models
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Import all the models, so that Base has them before being imported by Alembic
from app.db.base_class import Base  # noqa
from app.models.user import User  # noqa
from app.models.portfolio import Portfolio  # noqa
from app.models.stock import Stock  # noqa
from app.models.recommendation import Recommendation  # noqa
from app.models.sentiment_analysis import SentimentAnalysis  # noqa
from app.models.model_prediction import ModelPrediction  # noqa
from app.models.rag_document import RAGDocument  # noqa 