"""
Structured access-log middleware.

Emits one JSON log line per request containing method, path, status,
duration, and the bound request_id from structlog's context vars.
Replaces uvicorn's access log (which is disabled in CMD).
"""
from __future__ import annotations

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger("avengine.access")

# Paths excluded from access logging to avoid noise
_SILENT_PATHS = frozenset({"/health", "/health/ready", "/metrics"})


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path in _SILENT_PATHS:
            return await call_next(request)

        start = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            log = logger.info if status_code < 500 else logger.error
            log(
                "http.request",
                method=request.method,
                path=request.url.path,
                status=status_code,
                duration_ms=round(duration_ms, 2),
                client=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
