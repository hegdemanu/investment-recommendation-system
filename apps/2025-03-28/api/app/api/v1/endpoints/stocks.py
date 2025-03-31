from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_stocks():
    return {"stocks": [], "market_status": "open"} 