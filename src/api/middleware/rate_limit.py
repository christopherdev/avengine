"""
Sliding-window rate limiter backed by Redis.

Uses the INCR + EXPIRE pattern for atomicity with a Lua script so the
check-and-increment is a single round trip to Redis.

Default: 60 requests / 60 seconds per IP for the public API.
Authenticated clients can raise the limit by providing an API key
(resolved in the dependency layer — this middleware only handles IP-based limits).
"""
from __future__ import annotations

import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = structlog.get_logger(__name__)

# Lua script: atomically increment and optionally set TTL
_LUA_INCR = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, window)
end
return current
"""

# Paths exempt from rate limiting
_EXEMPT = frozenset({"/health", "/health/ready", "/docs", "/openapi.json"})


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: object,
        *,
        limit: int = 60,
        window_seconds: int = 60,
    ) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._limit = limit
        self._window = window_seconds

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path in _EXEMPT:
            return await call_next(request)

        redis = getattr(request.app.state, "redis", None)
        if redis is None:
            # Redis unavailable — fail open rather than blocking all traffic
            logger.warning("rate_limit.redis_unavailable")
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        bucket = f"rl:{client_ip}:{int(time.time()) // self._window}"

        script = redis.register_script(_LUA_INCR)
        current: int = await script(keys=[bucket], args=[self._limit, self._window])

        remaining = max(0, self._limit - current)
        reset_ts = (int(time.time()) // self._window + 1) * self._window

        if current > self._limit:
            logger.warning(
                "rate_limit.exceeded",
                client_ip=client_ip,
                path=request.url.path,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "type": "rate_limit_exceeded",
                    "title": "Too Many Requests",
                    "status": 429,
                    "detail": (
                        f"Rate limit of {self._limit} requests per "
                        f"{self._window}s exceeded."
                    ),
                },
                headers={
                    "Retry-After": str(reset_ts - int(time.time())),
                    "X-RateLimit-Limit": str(self._limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_ts),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_ts)
        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        # Honour X-Forwarded-For set by ALB / CloudFront
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
