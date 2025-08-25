from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class UserBase(BaseModel):
    name: str
    phone: Optional[str] = None


class UserCreate(UserBase):
    pass


class UserUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None


class User(UserBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
