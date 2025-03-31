"""
Database optimization utilities for the investment recommendation system.

This module provides utilities for database optimization, including
connection pooling, caching, and query optimization.
"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast
import time

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

from app.config import DATABASE_URL

logger = logging.getLogger(__name__)

# Create async engine with connection pooling optimized for API workloads
engine = create_async_engine(
    "sqlite+aiosqlite:///./app.db",  # Note the +aiosqlite driver specification
    echo=True,
    pool_size=5,              # Reasonable default for concurrent API requests
    max_overflow=10,          # Allow for traffic spikes
    pool_timeout=30,          # Timeout for getting a connection from the pool
    pool_recycle=3600,        # Recycle connections every hour to avoid staleness
    pool_pre_ping=True,       # Verify connection is still active before using
    future=True
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Simple in-memory query cache
query_cache: Dict[str, Dict[str, Any]] = {}

# Type variable for generic function return typing
T = TypeVar('T')

async def get_db() -> AsyncSession:
    """
    Get a database session with connection from the pool.
    
    Returns:
        AsyncSession: Database session
    """
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()

def cache_query(ttl_seconds: int = 300):
    """
    Cache decorator for database queries.
    
    Args:
        ttl_seconds: Time to live for cached results in seconds
    
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Create a cache key from function name and arguments
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Check if result is in cache and not expired
            if cache_key in query_cache:
                result_time = query_cache[cache_key].get("time", 0)
                if time.time() - result_time < ttl_seconds:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cast(T, query_cache[cache_key]["result"])
            
            # Execute the function and cache the result
            result = await func(*args, **kwargs)
            query_cache[cache_key] = {
                "result": result,
                "time": time.time()
            }
            return result
        return wrapper
    return decorator

async def optimize_database_tables(db: AsyncSession) -> None:
    """
    Perform database optimization tasks on all tables.
    For SQLite, this performs VACUUM to reclaim space and optimize indices.
    
    Args:
        db: Database session
    """
    if DATABASE_URL.startswith("sqlite"):
        await db.execute(text("VACUUM;"))
        await db.execute(text("PRAGMA optimize;"))
        logger.info("SQLite database optimized")
    elif DATABASE_URL.startswith("postgresql"):
        # PostgreSQL optimization
        await db.execute(text("VACUUM ANALYZE;"))
        logger.info("PostgreSQL database optimized")

async def create_indices(db: AsyncSession) -> None:
    """
    Create useful indices on the database tables.
    
    Args:
        db: Database session
    """
    # Add indices for common queries
    await db.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks (symbol);
    """))
    await db.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices (date);
    """))
    await db.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stock_prices (symbol, date);
    """))
    await db.commit()
    logger.info("Database indices created or already exist")

def clear_cache() -> None:
    """Clear the entire query cache."""
    query_cache.clear()
    logger.info("Query cache cleared") 