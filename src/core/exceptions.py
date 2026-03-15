"""
Domain-specific exception hierarchy.

All exceptions carry a machine-readable `code` so clients can handle them
programmatically without parsing human-readable messages.
"""
from __future__ import annotations

from http import HTTPStatus


class AVEngineError(Exception):
    """Base exception for all AVEngine errors."""

    http_status: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "avengine_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# ── Task / Pipeline ───────────────────────────────────────────────────────────

class TaskNotFoundError(AVEngineError):
    http_status = HTTPStatus.NOT_FOUND
    code = "task_not_found"


class TaskAlreadyExistsError(AVEngineError):
    http_status = HTTPStatus.CONFLICT
    code = "task_already_exists"


class PipelineError(AVEngineError):
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY
    code = "pipeline_error"


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentError(AVEngineError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "agent_error"


class ScriptGenerationError(AgentError):
    code = "script_generation_error"


class TimelineExtractionError(AgentError):
    code = "timeline_extraction_error"


# ── Sourcing ──────────────────────────────────────────────────────────────────

class SourcingError(AVEngineError):
    http_status = HTTPStatus.BAD_GATEWAY
    code = "sourcing_error"


class DownloadError(SourcingError):
    code = "download_error"


class CrawlerError(SourcingError):
    code = "crawler_error"


# ── Matching ──────────────────────────────────────────────────────────────────

class MatchingError(AVEngineError):
    http_status = HTTPStatus.UNPROCESSABLE_ENTITY
    code = "matching_error"


class EmbeddingError(MatchingError):
    code = "embedding_error"


# ── Rendering ─────────────────────────────────────────────────────────────────

class RenderingError(AVEngineError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "rendering_error"


class FFmpegError(RenderingError):
    code = "ffmpeg_error"


# ── External Services ─────────────────────────────────────────────────────────

class TwelveLabsError(AVEngineError):
    http_status = HTTPStatus.BAD_GATEWAY
    code = "twelvelabs_error"


class ElevenLabsError(AVEngineError):
    http_status = HTTPStatus.BAD_GATEWAY
    code = "elevenlabs_error"


class StorageError(AVEngineError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "storage_error"
