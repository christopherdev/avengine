"""
Authentication router.

  POST  /auth/login   → issue JWT access token
  GET   /auth/me      → return current user info (JWT required)
"""
from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import ORJSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user
from src.core.database import get_session
from src.core.models import User
from src.core.security import create_access_token, hash_password, verify_password
from src.schemas.auth import ChangePasswordRequest, LoginRequest, TokenResponse, UserResponse

router = APIRouter(prefix="/auth", tags=["Auth"])
logger = structlog.get_logger(__name__)

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.post(
    "/login",
    response_model=TokenResponse,
    response_class=ORJSONResponse,
    summary="Obtain a JWT access token",
)
async def login(body: LoginRequest, session: SessionDep) -> TokenResponse:
    from src.core.config import get_settings

    settings = get_settings()

    stmt = select(User).where(User.username == body.username)
    user: User | None = (await session.execute(stmt)).scalars().first()

    if user is None or not verify_password(body.password, user.hashed_password):
        logger.warning("auth.login_failed", username=body.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled.",
        )

    token = create_access_token(
        subject=user.id,
        extra={"username": user.username, "role": user.role},
        secret=settings.effective_jwt_secret,
        algorithm=settings.jwt_algorithm,
        expires_minutes=settings.jwt_expire_minutes,
    )

    logger.info("auth.login_success", username=user.username, role=user.role)
    return TokenResponse(
        access_token=token,
        user_id=user.id,
        username=user.username,
        role=user.role,
    )


@router.post(
    "/change-password",
    status_code=204,
    summary="Change the current user's password",
)
async def change_password(
    body: ChangePasswordRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    session: SessionDep,
) -> None:
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )

    # Re-fetch user in this session so we can update it
    user = await session.get(User, current_user.id)
    user.hashed_password = hash_password(body.new_password)
    logger.info("auth.password_changed", username=current_user.username)


@router.get(
    "/me",
    response_model=UserResponse,
    response_class=ORJSONResponse,
    summary="Return the currently authenticated user",
)
async def me(current_user: Annotated[User, Depends(get_current_user)]) -> UserResponse:
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )
