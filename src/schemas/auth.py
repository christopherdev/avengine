"""
Auth and user-management request / response schemas.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str
    role: str


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    role: Literal["admin", "user"] = "user"
    is_active: bool = True


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    password: str | None = Field(default=None, min_length=8)
    role: Literal["admin", "user"] | None = None
    is_active: bool | None = None
    # None = remove limit; absent = keep existing (checked via model_fields_set)
    daily_limit: int | None = Field(default=None, ge=1)


class UserResponse(BaseModel):
    id: str
    username: str
    role: str
    is_active: bool
    daily_limit: int | None = None
    created_at: datetime
