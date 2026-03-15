"""
Application lifespan event handlers.

Manages startup / teardown of all long-lived resources:
  - Database connection pool
  - Redis connection pool
  - Qdrant client
  - OpenTelemetry tracer provider
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan context manager — replaces on_event decorators."""
    await _startup(app)
    yield
    await _shutdown(app)


async def _startup(app: FastAPI) -> None:
    from src.core.config import get_settings
    from src.core.logging import configure_logging

    settings = get_settings()
    configure_logging(
        log_level=settings.log_level.value,
        json_logs=settings.is_production,
    )

    log = structlog.get_logger(__name__)
    log.info("avengine.startup", environment=settings.environment.value)

    # ── Database ──────────────────────────────────────────────────────────────
    from src.core.database import create_engine, run_migrations

    engine = await create_engine(settings.database_url)
    app.state.db_engine = engine
    await run_migrations()
    log.info("avengine.db.ready")

    # ── Redis ─────────────────────────────────────────────────────────────────
    import redis.asyncio as aioredis

    redis = aioredis.from_url(
        str(settings.redis_url),
        max_connections=settings.redis_max_connections,
        decode_responses=True,
    )
    app.state.redis = redis
    log.info("avengine.redis.ready")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    from qdrant_client import AsyncQdrantClient

    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
    app.state.qdrant = qdrant
    log.info("avengine.qdrant.ready")

    # ── OpenTelemetry ─────────────────────────────────────────────────────────
    if settings.enable_tracing:
        _configure_otel(settings)
        log.info("avengine.otel.ready")

    # ── Scratch directories ───────────────────────────────────────────────────
    import pathlib

    pathlib.Path(settings.video_output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(settings.video_scratch_dir).mkdir(parents=True, exist_ok=True)

    # ── Sentry ────────────────────────────────────────────────────────────────
    if settings.sentry_dsn:
        import sentry_sdk

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment.value,
            traces_sample_rate=0.2,
        )
        log.info("avengine.sentry.ready")

    # ── Bootstrap admin user ──────────────────────────────────────────────────
    await _bootstrap_admin(settings)

    log.info("avengine.startup.complete")


async def _shutdown(app: FastAPI) -> None:
    log = structlog.get_logger(__name__)
    log.info("avengine.shutdown")

    if engine := getattr(app.state, "db_engine", None):
        await engine.dispose()

    if redis := getattr(app.state, "redis", None):
        await redis.aclose()

    if qdrant := getattr(app.state, "qdrant", None):
        await qdrant.close()

    log.info("avengine.shutdown.complete")


async def _bootstrap_admin(settings: object) -> None:
    """Create the default admin user if no users exist in the database."""
    from sqlalchemy import func, select
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.core.database import _session_factory
    from src.core.models import User
    from src.core.security import hash_password

    log = structlog.get_logger(__name__)

    if _session_factory is None:
        return

    sf: async_sessionmaker[AsyncSession] = _session_factory
    async with sf() as session:
        count: int = (
            await session.execute(select(func.count()).select_from(User))
        ).scalar_one()

        if count > 0:
            return  # users already exist

        from ulid import ULID

        admin = User(
            id=str(ULID()),
            username=settings.admin_username,  # type: ignore[attr-defined]
            hashed_password=hash_password(settings.admin_password),  # type: ignore[attr-defined]
            role="admin",
            is_active=True,
        )
        session.add(admin)
        await session.commit()

    log.warning(
        "avengine.admin_bootstrapped",
        username=settings.admin_username,  # type: ignore[attr-defined]
        note="Change the default admin password immediately in production.",
    )


def _configure_otel(settings: object) -> None:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create({"service.name": "avengine"})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint)  # type: ignore[attr-defined]
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
