from fastapi import APIRouter
from src.api.v1.endpoints import auth, portfolio, recommendation

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(portfolio.router, prefix="/portfolios", tags=["portfolios"])
api_router.include_router(recommendation.router, prefix="/recommendations", tags=["recommendations"])