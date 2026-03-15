"""
Structured logging configuration using structlog.

Outputs JSON in production, colourized console output in development.
Integrates with OpenTelemetry trace/span IDs for distributed tracing correlation.
"""
from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

# ── Sensitive-value redaction ──────────────────────────────────────────────────

# Patterns that look like secrets in log field *values*
_REDACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}"),        # OpenAI / Anthropic keys
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}"),    # Anthropic key prefix variant
    re.compile(r"(?i)(password|passwd|secret)[=: ]+\S+"),  # password= / secret=
    re.compile(r"(?i)bearer [A-Za-z0-9\-_.~+/]+=*"),      # Bearer tokens
]
_REDACTED = "[REDACTED]"


def _redact_value(value: str) -> str:
    for pattern in _REDACT_PATTERNS:
        value = pattern.sub(_REDACTED, value)
    return value


def _scrub_secrets(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """
    Walk every string field in the log event and redact known secret patterns.

    This is a defence-in-depth measure — code should never log secrets in the
    first place, but this processor prevents accidental disclosure when it does.
    """
    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            scrubbed = _redact_value(value)
            if scrubbed != value:
                event_dict[key] = scrubbed
    return event_dict


def _add_app_context(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Inject static app metadata into every log record."""
    from src.core.config import get_settings

    settings = get_settings()
    event_dict["app"] = settings.app_name
    event_dict["env"] = settings.environment.value
    return event_dict


def _extract_otel_context(
    logger: WrappedLogger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: EventDict,
) -> EventDict:
    """Inject OpenTelemetry trace_id / span_id when tracing is active."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
    except ImportError:
        pass
    return event_dict


def configure_logging(log_level: str = "INFO", *, json_logs: bool = True) -> None:
    """
    Bootstrap structlog and stdlib logging.

    Call once at application startup (before any log statements).
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_app_context,
        _extract_otel_context,
        structlog.processors.StackInfoRenderer(),
        _scrub_secrets,
    ]

    if json_logs:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Silence noisy third-party loggers
    for name in ("uvicorn.access", "celery.app.trace", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
