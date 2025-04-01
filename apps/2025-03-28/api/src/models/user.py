from typing import Optional
from sqlmodel import SQLModel, Field, Relationship
from pydantic import EmailStr, constr
from src.models.base import BaseModel

class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True)
    username: str = Field(unique=True, index=True)
    full_name: str
    risk_tolerance: Optional[int] = Field(default=None)
    investment_horizon: Optional[int] = Field(default=None)  # in months
    investment_goal: Optional[str] = Field(default=None)
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)

class User(BaseModel, UserBase, table=True):
    hashed_password: str
    portfolios: list["Portfolio"] = Relationship(back_populates="user")

class UserCreate(UserBase):
    password: constr(min_length=8)

class UserUpdate(SQLModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[constr(min_length=8)] = None
    risk_tolerance: Optional[int] = None
    investment_horizon: Optional[int] = None
    investment_goal: Optional[str] = None
    is_active: Optional[bool] = None

class UserResponse(UserBase):
    id: str

    class Config:
        orm_mode = True 