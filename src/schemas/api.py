"""
Public-facing API request / response schemas.

These are the contracts the HTTP layer exposes to callers.
Domain-internal schemas (VideoTimeline, ScriptScene, etc.) live in
schemas/pipeline.py and are authored in Step 3.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class VideoStyle(str, Enum):
    documentary = "documentary"
    explainer = "explainer"
    social_short = "social_short"
    news_recap = "news_recap"
    tutorial = "tutorial"
    reels = "reels"


class AspectRatio(str, Enum):
    landscape = "16:9"
    portrait = "9:16"
    square = "1:1"


# ── Request Bodies ─────────────────────────────────────────────────────────────

class GenerateVideoRequest(BaseModel):
    """
    Payload for POST /generate-video.

    The engine turns this into the initial LangGraph state and dispatches
    a Celery pipeline task, returning 202 immediately.
    """

    topic: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Subject matter or brief for the video.",
        examples=["The rise of autonomous AI agents in 2025"],
    )
    style: VideoStyle = Field(
        default=VideoStyle.explainer,
        description="Tone and format of the generated video.",
    )
    duration_seconds: int = Field(
        default=60,
        ge=15,
        le=300,
        description="Target video length in seconds (15–300).",
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.landscape,
        description="Output aspect ratio.",
    )
    target_audience: str = Field(
        default="general",
        max_length=200,
        description="Intended audience — used to tune script tone.",
        examples=["software engineers", "high school students"],
    )
    seed_urls: list[AnyHttpUrl] = Field(
        default_factory=list,
        max_length=10,
        description="Optional URLs the Sourcing agent should prioritise.",
    )
    voice_id: str | None = Field(
        default=None,
        description="ElevenLabs voice ID override. Falls back to settings default.",
    )
    webhook_url: AnyHttpUrl | None = Field(
        default=None,
        description=(
            "If provided, AVEngine POSTs the TaskResult JSON here when the "
            "pipeline finishes (success or failure)."
        ),
    )

    @model_validator(mode="after")
    def _reels_defaults(self) -> "GenerateVideoRequest":
        """Reels style implies portrait 9:16 and a short duration cap."""
        if self.style == VideoStyle.reels:
            if self.aspect_ratio == AspectRatio.landscape:
                self.aspect_ratio = AspectRatio.portrait
            if self.duration_seconds > 60:
                self.duration_seconds = 60
        return self

    @field_validator("seed_urls", mode="before")
    @classmethod
    def _coerce_urls(cls, v: Any) -> list[Any]:
        if isinstance(v, str):
            return [v]
        return v or []


# ── Task Event (emitted on the Pub/Sub bus + WebSocket) ───────────────────────

class AgentStage(str, Enum):
    ideation = "ideation"
    scripting = "scripting"
    sourcing = "sourcing"
    matching = "matching"
    rendering = "rendering"
    done = "done"
    error = "error"


class TaskEvent(BaseModel):
    """
    Envelope published to Redis and forwarded over WebSocket.

    type mirrors AgentStage so clients can drive progress UIs.
    """

    task_id: str
    type: AgentStage
    message: str = ""
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"use_enum_values": True}


# ── Response Bodies ───────────────────────────────────────────────────────────

class AcceptedResponse(BaseModel):
    """202 Accepted envelope returned by POST /generate-video."""

    task_id: str = Field(description="ULID task identifier.")
    status: TaskStatus = TaskStatus.queued
    status_url: str = Field(description="Polling URL for task status.")
    ws_url: str = Field(description="WebSocket URL for real-time events.")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskStatusResponse(BaseModel):
    """GET /tasks/{task_id} — polling response."""

    task_id: str
    topic: str | None = None
    style: str | None = None
    status: TaskStatus
    stage: AgentStage | None = None
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    created_at: datetime
    updated_at: datetime
    error: str | None = None


class TaskResult(BaseModel):
    """GET /tasks/{task_id}/result — final artefact URLs."""

    task_id: str
    status: TaskStatus
    video_url: str | None = None
    thumbnail_url: str | None = None
    script: str | None = None
    duration_seconds: float | None = None
    completed_at: datetime | None = None
    error: str | None = None


class TaskListResponse(BaseModel):
    items: list[TaskStatusResponse]
    total: int
    page: int
    page_size: int


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthStatus(str, Enum):
    healthy = "healthy"
    degraded = "degraded"
    unhealthy = "unhealthy"


class ComponentHealth(BaseModel):
    status: HealthStatus
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: HealthStatus
    version: str
    environment: str
    components: dict[str, ComponentHealth] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Generic error envelope (matches RFC 7807 Problem Details) ─────────────────

class ErrorDetail(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    instance: str | None = None
    trace_id: str | None = None
