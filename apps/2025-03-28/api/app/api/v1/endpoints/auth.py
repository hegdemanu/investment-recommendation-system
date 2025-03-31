from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter()

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    return {
        "access_token": "dummy_token",
        "token_type": "bearer"
    }

@router.post("/test-connection")
async def test_connection():
    """
    Test endpoint to verify API is working.
    """
    return {"status": "success", "message": "API is working correctly"} 