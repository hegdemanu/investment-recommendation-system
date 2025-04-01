from app.models.user import User
from app.models.portfolio import Portfolio, Investment

# For Alembic auto-generation of migrations
__all__ = ["User", "Portfolio", "Investment"]

"""
Models package for the investment recommendation system.

This package contains database models, schemas, and other data structures.
"""
