"""
Async SQLAlchemy engine + session factory.

Uses asyncpg driver. Sessions are provided via FastAPI dependency injection.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


class Base(DeclarativeBase):
    """SQLAlchemy declarative base shared by all ORM models."""


async def create_engine(database_url: str) -> AsyncEngine:
    global _engine, _session_factory  # noqa: PLW0603

    from src.core.config import get_settings

    settings = get_settings()
    _engine = create_async_engine(
        database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_pre_ping=True,
        echo=settings.debug,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    return _engine


async def run_migrations() -> None:
    """Run Alembic migrations programmatically at startup."""
    import asyncio
    import os
    from functools import partial

    from alembic import command
    from alembic.config import Config

    from src.core.config import get_settings

    settings = get_settings()
    # alembic/env.py reads DATABASE_URL from os.getenv(); pydantic-settings does
    # not populate os.environ, so we inject it here for the alembic thread.
    os.environ.setdefault("DATABASE_URL", settings.database_url)

    alembic_cfg = Config("alembic.ini")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, partial(command.upgrade, alembic_cfg, "head"))


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields a database session per request."""
    if _session_factory is None:
        raise RuntimeError("Database not initialised. Call create_engine() first.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
