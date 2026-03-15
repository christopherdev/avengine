"""
Health check endpoints.

  GET /health        — liveness probe (always returns 200 if the process is up)
  GET /health/ready  — readiness probe (validates DB, Redis, Qdrant connectivity)

ECS / ALB target-group health checks hit /health.
Kubernetes liveness/readiness probes use the same pattern.
"""
from __future__ import annotations

import asyncio
import time

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import ORJSONResponse

from src.core.config import get_settings
from src.schemas.api import ComponentHealth, HealthResponse, HealthStatus

router = APIRouter(tags=["Health"])
logger = structlog.get_logger(__name__)
settings = get_settings()


@router.get(
    "/health",
    response_model=HealthResponse,
    response_class=ORJSONResponse,
    summary="Liveness probe",
)
async def liveness() -> HealthResponse:
    """Returns 200 immediately — confirms the process is alive."""
    return HealthResponse(
        status=HealthStatus.healthy,
        version="0.1.0",
        environment=settings.environment.value,
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    response_class=ORJSONResponse,
    summary="Readiness probe",
)
async def readiness(request: Request) -> ORJSONResponse:
    """
    Checks all downstream dependencies in parallel.
    Returns 200 if all are healthy, 503 if any are degraded/unhealthy.
    """
    checks = await asyncio.gather(
        _check_postgres(request),
        _check_redis(request),
        _check_qdrant(request),
        return_exceptions=True,
    )

    components = {}
    for label, result in zip(("postgres", "redis", "qdrant"), checks, strict=True):
        if isinstance(result, Exception):
            components[label] = ComponentHealth(
                status=HealthStatus.unhealthy,
                detail=str(result),
            )
        else:
            components[label] = result

    overall = (
        HealthStatus.healthy
        if all(c.status == HealthStatus.healthy for c in components.values())
        else HealthStatus.unhealthy
    )

    body = HealthResponse(
        status=overall,
        version="0.1.0",
        environment=settings.environment.value,
        components=components,
    )
    http_status = 200 if overall == HealthStatus.healthy else 503
    return ORJSONResponse(content=body.model_dump(mode="json"), status_code=http_status)


# ── Component checks ──────────────────────────────────────────────────────────

async def _check_postgres(request: Request) -> ComponentHealth:
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        return ComponentHealth(status=HealthStatus.unhealthy, detail="engine not initialised")
    start = time.perf_counter()
    try:
        async with engine.connect() as conn:
            await conn.execute_fetchall("SELECT 1")  # type: ignore[attr-defined]
        return ComponentHealth(
            status=HealthStatus.healthy,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth(status=HealthStatus.unhealthy, detail=str(exc))


async def _check_redis(request: Request) -> ComponentHealth:
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return ComponentHealth(status=HealthStatus.unhealthy, detail="redis not initialised")
    start = time.perf_counter()
    try:
        await redis.ping()
        return ComponentHealth(
            status=HealthStatus.healthy,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth(status=HealthStatus.unhealthy, detail=str(exc))


async def _check_qdrant(request: Request) -> ComponentHealth:
    qdrant = getattr(request.app.state, "qdrant", None)
    if qdrant is None:
        return ComponentHealth(status=HealthStatus.unhealthy, detail="qdrant not initialised")
    start = time.perf_counter()
    try:
        await qdrant.get_collections()
        return ComponentHealth(
            status=HealthStatus.healthy,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
    except Exception as exc:  # noqa: BLE001
        return ComponentHealth(status=HealthStatus.degraded, detail=str(exc))
