"""
AVEngine — FastAPI application factory.

Usage:
  uvicorn src.main:app --reload               # development
  uvicorn src.main:app --workers 4            # production (single-process)
  gunicorn -k uvicorn.workers.UvicornWorker   # multi-process

The `app` object is the ASGI application.  All wiring (middleware, routers,
exception handlers, lifespan) is assembled here and only here.
"""
from __future__ import annotations

from http import HTTPStatus
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.staticfiles import StaticFiles

from src.core.config import get_settings
from src.core.events import lifespan
from src.core.exceptions import AVEngineError
from src.schemas.api import ErrorDetail

settings = get_settings()
logger = structlog.get_logger(__name__)


# ── Application Factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Construct and return the configured FastAPI application.

    Separated from module-level `app` instantiation so tests can call
    `create_app()` with overridden settings without polluting global state.
    """
    application = FastAPI(
        title="AVEngine",
        description=(
            "Autonomous Multi-Agent Video Creation Engine — "
            "LangGraph + CrewAI + TwelveLabs + MoviePy"
        ),
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    # Store prefix on app state so routers can build absolute URLs
    application.state.api_prefix = settings.api_prefix

    _add_middleware(application)
    _add_exception_handlers(application)
    _add_routers(application)

    return application


# ── Middleware Stack (order matters — outermost first) ────────────────────────

def _add_middleware(app: FastAPI) -> None:
    # 1. Security headers — outermost so every response (incl. error handlers)
    #    gets the defensive headers applied.
    from src.api.middleware.security_headers import SecurityHeadersMiddleware

    app.add_middleware(SecurityHeadersMiddleware, is_production=settings.is_production)

    # 2. Trusted Host — rejects requests with forged Host headers
    if settings.allowed_hosts != ["*"]:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts,
        )

    # 3. CORS — wildcard only when CORS_ORIGINS is not explicitly set.
    #    ALWAYS set CORS_ORIGINS in production to the specific frontend domain.
    cors_origins = [str(o) for o in settings.cors_origins]
    if not cors_origins and settings.is_production:
        logger.warning(
            "cors.wildcard_in_production",
            note="Set CORS_ORIGINS env var to restrict allowed origins.",
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        # Explicit method + header lists instead of "*" to reduce attack surface
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining"],
    )

    # 4. Rate limiting (Redis-backed sliding window)
    from src.api.middleware.rate_limit import RateLimitMiddleware

    app.add_middleware(RateLimitMiddleware, limit=60, window_seconds=60)

    # 5. Structured access logging
    from src.api.middleware.logging import AccessLogMiddleware

    app.add_middleware(AccessLogMiddleware)

    # 6. Request-ID injection (innermost — must run before logging)
    from src.api.middleware.request_id import RequestIDMiddleware

    app.add_middleware(RequestIDMiddleware)


# ── Exception Handlers ────────────────────────────────────────────────────────

def _add_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(AVEngineError)
    async def _handle_domain_error(request: Request, exc: AVEngineError) -> ORJSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "domain_error",
            code=exc.code,
            message=exc.message,
            path=str(request.url),
        )
        body = ErrorDetail(
            type=f"https://avengine.dev/errors/{exc.code}",
            title=exc.code.replace("_", " ").title(),
            status=exc.http_status,
            detail=exc.message,
            instance=str(request.url),
            trace_id=request_id,
        )
        return ORJSONResponse(
            status_code=exc.http_status,
            content=body.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> ORJSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.info("validation_error", errors=exc.errors(), path=str(request.url))
        body = ErrorDetail(
            type="https://avengine.dev/errors/validation_error",
            title="Validation Error",
            status=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=str(exc.errors()),
            instance=str(request.url),
            trace_id=request_id,
        )
        return ORJSONResponse(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            content=body.model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def _handle_unhandled(request: Request, exc: Exception) -> ORJSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception("unhandled_error", path=str(request.url))
        body = ErrorDetail(
            type="https://avengine.dev/errors/internal_server_error",
            title="Internal Server Error",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred.",
            instance=str(request.url),
            trace_id=request_id,
        )
        return ORJSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content=body.model_dump(mode="json"),
        )


# ── Router Registration ───────────────────────────────────────────────────────

def _add_routers(app: FastAPI) -> None:
    import pathlib

    from fastapi import Depends

    from src.api.dependencies.auth import require_admin, verify_api_key
    from src.api.routers.auth import router as auth_router
    from src.api.routers.health import router as health_router
    from src.api.routers.users import router as users_router
    from src.api.routers.video import router as video_router
    from src.api.routers.video import tasks_router
    from src.api.routers.ws import router as ws_router

    _auth = [Depends(verify_api_key)]

    # Health (no prefix — ALB hits /health directly; no auth required)
    app.include_router(health_router)

    # Auth (login / me) — no global auth guard; individual endpoints handle it
    app.include_router(auth_router, prefix=settings.api_prefix)

    # User management — admin-only at router level (defence in depth)
    app.include_router(
        users_router,
        prefix=settings.api_prefix,
        dependencies=[Depends(require_admin)],
    )

    # Versioned API — X-API-Key or Bearer JWT required
    app.include_router(video_router, prefix=settings.api_prefix, dependencies=_auth)
    app.include_router(tasks_router, prefix=settings.api_prefix, dependencies=_auth)
    # WebSocket — no router-level auth dependency; browsers cannot send custom
    # headers during the WS upgrade.  The task_id in the URL provides implicit
    # scoping (non-existent task_ids return an error snapshot).
    app.include_router(ws_router, prefix=settings.api_prefix)

    # Video output — served at /output/<task_id>/output.mp4
    output_dir = pathlib.Path(settings.video_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
         logger.error("output_dir.missing", path=str(output_dir))
    app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")

    # Frontend — served at /ui (index.html at /ui/index.html)
    frontend_dir = pathlib.Path(__file__).parent.parent / "frontend"
    if frontend_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ── Module-level app instance (used by uvicorn / gunicorn) ────────────────────
app = create_app()


# ── Dev entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_config=None,      # structlog owns logging
        loop="uvloop",
        http="httptools",
    )


if __name__ == "__main__":
    main()
