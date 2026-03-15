"""
Rotating Residential Proxy Pool.

Maintains a priority-queue of proxy endpoints ordered by health score.
Each extraction attempt checks out a proxy, and the result (success/failure)
feeds back into the score so bad proxies are deprioritised automatically.

Proxy URL format: http://user:pass@host:port
Credentials can be parameterised using session IDs for sticky sessions:
  http://user-session-{session_id}:pass@host:port

Two operational modes are supported:
  1. Single endpoint with username-parameterised session IDs (e.g. Bright Data,
     Oxylabs, Smartproxy) — set PROXY_ENDPOINT + PROXY_USERNAME + PROXY_PASSWORD.
  2. Explicit pool list loaded from PROXY_POOL_JSON env var (JSON array of URLs).
"""
from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

import structlog

from src.core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

_HEALTH_DECAY = 0.15     # score penalty per failure
_HEALTH_RECOVER = 0.05   # score recovery per success
_MIN_HEALTH = 0.0
_MAX_HEALTH = 1.0
_QUARANTINE_SECONDS = 120


@dataclass
class ProxyEntry:
    url: str
    health: float = field(default=1.0)
    last_used: float = field(default=0.0)
    failures: int = field(default=0)
    quarantined_until: float = field(default=0.0)

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.quarantined_until

    def record_success(self) -> None:
        self.health = min(_MAX_HEALTH, self.health + _HEALTH_RECOVER)
        self.failures = max(0, self.failures - 1)
        self.last_used = time.monotonic()

    def record_failure(self) -> None:
        self.health = max(_MIN_HEALTH, self.health - _HEALTH_DECAY)
        self.failures += 1
        self.last_used = time.monotonic()
        if self.failures >= 3:
            self.quarantined_until = time.monotonic() + _QUARANTINE_SECONDS
            logger.warning("proxy_pool.quarantined", url=self._safe_url, failures=self.failures)

    @property
    def _safe_url(self) -> str:
        """URL with password redacted for logging."""
        try:
            from urllib.parse import urlparse

            p = urlparse(self.url)
            return p._replace(netloc=f"{p.username}:***@{p.hostname}:{p.port}").geturl()
        except Exception:  # noqa: BLE001
            return "<proxy>"


class ProxyPool:
    """
    Thread-safe rotating proxy pool.

    Usage:
        pool = ProxyPool.from_settings()
        async with pool.checkout() as proxy_url:
            # use proxy_url for the request
    """

    def __init__(self, proxies: list[str]) -> None:
        self._entries: list[ProxyEntry] = [ProxyEntry(url=p) for p in proxies]
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(cls) -> ProxyPool:
        """
        Build pool from application settings.

        Priority:
          1. PROXY_POOL_JSON — explicit list of proxy URLs
          2. PROXY_ENDPOINT + credentials — generates 10 session-based URLs
        """
        import json
        import os

        pool_json = os.getenv("PROXY_POOL_JSON", "")
        if pool_json:
            try:
                urls: list[str] = json.loads(pool_json)
                logger.info("proxy_pool.loaded_from_json", count=len(urls))
                return cls(urls)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("proxy_pool.json_parse_failed", error=str(exc))

        if settings.proxy_endpoint and settings.proxy_username:
            urls = [
                f"http://{settings.proxy_username}-session-{random.randint(1000, 9999)}"
                f":{settings.proxy_password}@{settings.proxy_endpoint}"
                for _ in range(10)
            ]
            logger.info("proxy_pool.generated_session_pool", count=len(urls))
            return cls(urls)

        logger.warning("proxy_pool.no_proxies_configured")
        return cls([])

    @property
    def is_empty(self) -> bool:
        return len(self._entries) == 0

    def _select(self) -> ProxyEntry | None:
        """
        Pick the highest-health available proxy.
        Falls back to the least-recently-used quarantined proxy if all are quarantined.
        """
        available = [e for e in self._entries if e.available]
        if available:
            return max(available, key=lambda e: e.health)

        # All quarantined — pick the one whose quarantine expires soonest
        if self._entries:
            return min(self._entries, key=lambda e: e.quarantined_until)

        return None

    @asynccontextmanager
    async def checkout(self) -> AsyncIterator[str | None]:
        """
        Async context manager that:
          - Yields the best available proxy URL (or None if pool is empty)
          - Records success/failure back to the entry on context exit
        """
        async with self._lock:
            entry = self._select()

        if entry is None:
            yield None
            return

        try:
            yield entry.url
            async with self._lock:
                entry.record_success()
        except Exception:
            async with self._lock:
                entry.record_failure()
            raise

    def report_failure(self, proxy_url: str) -> None:
        """Explicitly report a proxy failure (for callers that catch exceptions)."""
        for entry in self._entries:
            if entry.url == proxy_url:
                entry.record_failure()
                break

    def stats(self) -> list[dict]:
        return [
            {
                "url": e._safe_url,
                "health": round(e.health, 2),
                "failures": e.failures,
                "available": e.available,
            }
            for e in self._entries
        ]


# Module-level singleton — shared within a worker process
_pool: ProxyPool | None = None


def get_proxy_pool() -> ProxyPool:
    global _pool  # noqa: PLW0603
    if _pool is None:
        _pool = ProxyPool.from_settings()
    return _pool
