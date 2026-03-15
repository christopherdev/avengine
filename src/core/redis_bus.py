"""
Async Redis Pub/Sub message bus.

Used by agents to broadcast state-change events, and by the WebSocket
endpoint to fan them out to connected clients in real time.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import redis.asyncio as aioredis
import structlog

from src.core.config import get_settings

logger = structlog.get_logger(__name__)

_TASK_CHANNEL_PREFIX = "task:"


def task_channel(task_id: str) -> str:
    return f"{_TASK_CHANNEL_PREFIX}{task_id}"


class RedisBus:
    """
    Thin wrapper around redis.asyncio pub/sub.

    One instance is shared application-wide (stored on app.state).
    """

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    async def publish(self, task_id: str, event: dict[str, Any]) -> None:
        channel = task_channel(task_id)
        payload = json.dumps(event, default=str)
        await self._redis.publish(channel, payload)
        logger.debug("redis_bus.published", channel=channel, event_type=event.get("type"))

    async def subscribe(self, task_id: str) -> AsyncIterator[dict[str, Any]]:
        """
        Async generator that yields parsed event dicts from a task channel.

        The generator exits when a terminal event (type == "done" | "error")
        is received or the connection is closed.
        """
        pubsub = self._redis.pubsub()
        channel = task_channel(task_id)
        await pubsub.subscribe(channel)
        logger.debug("redis_bus.subscribed", channel=channel)

        try:
            async for raw_message in pubsub.listen():
                if raw_message["type"] != "message":
                    continue
                try:
                    event: dict[str, Any] = json.loads(raw_message["data"])
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("redis_bus.decode_error", error=str(exc))
                    continue

                yield event

                if event.get("type") in {"done", "error"}:
                    break
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()


def get_redis_bus(app_state: Any) -> RedisBus:
    """Retrieve the RedisBus from FastAPI app state."""
    return RedisBus(app_state.redis)
