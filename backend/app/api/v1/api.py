from fastapi import APIRouter
from app.api.v1.endpoints import auth, users, portfolio, investments

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
api_router.include_router(investments.router, prefix="/investments", tags=["investments"]) 