"""
Authentication dependencies.

Three dependency levels are provided:

  verify_api_key   — accepts X-API-Key header OR Bearer JWT (any valid user)
  get_current_user — requires Bearer JWT; returns the User ORM object
  require_admin    — like get_current_user but also enforces role == "admin"

The video / tasks / ws routers use `verify_api_key` (backwards-compatible
with the previous X-API-Key-only scheme and allows browser JWT auth).

The users management router uses `require_admin`.
The /auth/me endpoint uses `get_current_user`.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.core.config import get_settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_BEARER = HTTPBearer(auto_error=False)


# ── JWT helpers ────────────────────────────────────────────────────────────────

async def _decode_bearer(
    credentials: HTTPAuthorizationCredentials | None,
) -> dict | None:
    """Return the decoded JWT payload or None if no/invalid Bearer token."""
    if credentials is None:
        return None
    from jose import JWTError

    from src.core.security import decode_token

    settings = get_settings()
    try:
        return decode_token(
            credentials.credentials,
            settings.effective_jwt_secret,
            settings.jwt_algorithm,
        )
    except JWTError:
        return None


# ── verify_api_key (X-API-Key OR Bearer JWT — any valid user) ─────────────────

async def verify_api_key(
    api_key: str | None = Security(_API_KEY_HEADER),
    credentials: HTTPAuthorizationCredentials | None = Security(_BEARER),
) -> str:
    """
    Validates the request carries either:
      • A valid X-API-Key header, or
      • A valid Bearer JWT

    When API_KEYS is empty *and* no valid JWT is present, the request is
    allowed in development (auth disabled) but rejected in production.
    """
    settings = get_settings()

    # 1. Try JWT Bearer first (preferred for browser clients)
    payload = await _decode_bearer(credentials)
    if payload is not None:
        return payload.get("sub", "jwt-user")

    # 2. Try X-API-Key
    valid_keys = settings.api_key_list
    if api_key and valid_keys and api_key in valid_keys:
        return api_key

    # 3. Explicit dev anonymous bypass — must be opted in via ALLOW_ANONYMOUS_DEV=true
    if settings.allow_anonymous_dev and not settings.is_production:
        return "anonymous"

    # 4. Reject
    if api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Supply X-API-Key or Bearer token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ── get_current_user (Bearer JWT required — returns User ORM object) ──────────

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_BEARER),
) -> "User":  # type: ignore[name-defined]  # noqa: F821
    """
    Requires a valid Bearer JWT.  Returns the corresponding User row.
    Raises 401 if missing/invalid, 401 if user not found or inactive.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = await _decode_bearer(credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str | None = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload.")

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from src.core.config import get_settings
    from src.core.models import User

    s = get_settings()
    engine = create_async_engine(s.database_url, poolclass=NullPool)
    sf = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with sf() as session:
        user = await session.get(User, user_id)
    await engine.dispose()

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
        )
    return user


# ── get_optional_current_user (returns User | None — never raises) ────────────

async def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_BEARER),
) -> "User | None":  # type: ignore[name-defined]  # noqa: F821
    """
    Like get_current_user but returns None instead of raising 401.
    Use for endpoints that behave differently for authenticated vs anonymous.
    """
    if credentials is None:
        return None
    payload = await _decode_bearer(credentials)
    if payload is None:
        return None
    user_id: str | None = payload.get("sub")
    if not user_id:
        return None

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from src.core.models import User

    s = get_settings()
    engine = create_async_engine(s.database_url, poolclass=NullPool)
    sf = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with sf() as session:
        user = await session.get(User, user_id)
    await engine.dispose()

    if user is None or not user.is_active:
        return None
    return user


# ── require_admin ─────────────────────────────────────────────────────────────

async def require_admin(
    user: Annotated["User", Depends(get_current_user)],  # type: ignore[name-defined]  # noqa: F821
) -> "User":  # type: ignore[name-defined]  # noqa: F821
    """Extends get_current_user — additionally enforces role == 'admin'."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required.",
        )
    return user
