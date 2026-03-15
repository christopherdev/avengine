"""
Pipeline → Redis event publisher.

Each graph node calls `publish_event()` to broadcast its progress to the
Redis pub/sub channel, which the WebSocket router fans out to connected
clients in real time.

This module is intentionally synchronous-first (using asyncio.run / a
dedicated event loop) because LangGraph nodes run in a synchronous Celery
worker context.  For async-native usage (e.g. tests), use `async_publish`.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import structlog

from src.schemas.api import AgentStage, TaskEvent

logger = structlog.get_logger(__name__)

# Module-level redis client — lazily initialised once per worker process
_redis_client: Any = None


def _get_redis() -> Any:
    global _redis_client  # noqa: PLW0603
    if _redis_client is None:
        import redis as sync_redis

        from src.core.config import get_settings

        settings = get_settings()
        _redis_client = sync_redis.from_url(
            str(settings.redis_url), decode_responses=True
        )
    return _redis_client


def publish_event(
    task_id: str,
    stage: AgentStage,
    message: str,
    progress: float,
    data: dict[str, Any] | None = None,
) -> None:
    """
    Synchronous publish — safe to call from a Celery worker.

    Uses the synchronous redis client to avoid running a nested event loop.
    """
    event = TaskEvent(
        task_id=task_id,
        type=stage,
        message=message,
        progress=progress,
        data=data or {},
        timestamp=datetime.now(tz=timezone.utc),
    )
    channel = f"task:{task_id}"
    payload = event.model_dump_json()

    try:
        _get_redis().publish(channel, payload)
        logger.debug("pipeline.event_published", task_id=task_id, stage=stage, progress=progress)
    except Exception as exc:  # noqa: BLE001
        # Non-fatal — the pipeline continues even if the pub/sub fails
        logger.warning("pipeline.publish_failed", task_id=task_id, error=str(exc))


async def async_publish(
    task_id: str,
    stage: AgentStage,
    message: str,
    progress: float,
    data: dict[str, Any] | None = None,
    redis: Any = None,
) -> None:
    """
    Async publish — for test contexts or async pipeline runners.
    """
    event = TaskEvent(
        task_id=task_id,
        type=stage,
        message=message,
        progress=progress,
        data=data or {},
        timestamp=datetime.now(tz=timezone.utc),
    )
    channel = f"task:{task_id}"
    payload = event.model_dump_json()

    if redis is None:
        import redis.asyncio as aioredis

        from src.core.config import get_settings

        settings = get_settings()
        redis = aioredis.from_url(str(settings.redis_url), decode_responses=True)

    await redis.publish(channel, payload)
