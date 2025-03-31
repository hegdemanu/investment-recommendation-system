from fastapi import APIRouter

router = APIRouter()

@router.get("/me")
async def read_user_me():
    return {"user_id": "current", "username": "testuser"} 