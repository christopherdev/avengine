"""
User management router (admin only).

  GET    /users              → list all users
  POST   /users              → create a new user
  PATCH  /users/{user_id}   → update role / active status / password
  DELETE /users/{user_id}   → deactivate (soft-delete) a user
"""
from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import ORJSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from src.api.dependencies.auth import require_admin
from src.core.database import get_session
from src.core.models import User
from src.core.security import hash_password
from src.schemas.auth import UserCreate, UserResponse, UserUpdate

router = APIRouter(prefix="/users", tags=["Users"])
logger = structlog.get_logger(__name__)

SessionDep = Annotated[AsyncSession, Depends(get_session)]
AdminDep = Annotated[User, Depends(require_admin)]


@router.get(
    "",
    response_model=list[UserResponse],
    response_class=ORJSONResponse,
    summary="List all users",
)
async def list_users(session: SessionDep, _admin: AdminDep) -> list[UserResponse]:
    rows = (await session.execute(select(User).order_by(User.created_at))).scalars().all()
    return [
        UserResponse(
            id=u.id,
            username=u.username,
            role=u.role,
            is_active=u.is_active,
            daily_limit=u.daily_limit,
            created_at=u.created_at,
        )
        for u in rows
    ]


@router.post(
    "",
    status_code=201,
    response_model=UserResponse,
    response_class=ORJSONResponse,
    summary="Create a new user",
)
async def create_user(
    body: UserCreate, session: SessionDep, admin: AdminDep
) -> UserResponse:
    # Check username uniqueness
    existing = (
        await session.execute(select(User).where(User.username == body.username))
    ).scalars().first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{body.username}' is already taken.",
        )

    user = User(
        id=str(ULID()),
        username=body.username,
        hashed_password=hash_password(body.password),
        role=body.role,
        is_active=body.is_active,
    )
    session.add(user)
    await session.flush()

    logger.info("user.created", username=user.username, role=user.role, by=admin.username)
    return UserResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        daily_limit=user.daily_limit,
        created_at=user.created_at,
    )


@router.patch(
    "/{user_id}",
    response_model=UserResponse,
    response_class=ORJSONResponse,
    summary="Update a user's role, active status, or password",
)
async def update_user(
    user_id: str, body: UserUpdate, session: SessionDep, admin: AdminDep
) -> UserResponse:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    if body.role is not None:
        user.role = body.role
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.password is not None:
        user.hashed_password = hash_password(body.password)
    if "daily_limit" in body.model_fields_set:
        user.daily_limit = body.daily_limit  # None removes the limit

    logger.info("user.updated", user_id=user_id, by=admin.username)
    return UserResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        daily_limit=user.daily_limit,
        created_at=user.created_at,
    )


@router.delete(
    "/{user_id}",
    status_code=204,
    summary="Deactivate a user",
)
async def delete_user(user_id: str, session: SessionDep, admin: AdminDep) -> None:
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    # Prevent admins from deactivating themselves
    if user.id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot deactivate your own account.",
        )

    user.is_active = False
    logger.info("user.deactivated", user_id=user_id, username=user.username, by=admin.username)
