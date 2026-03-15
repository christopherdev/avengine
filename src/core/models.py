"""
SQLAlchemy ORM models — persisted task state.

Tasks are the durable record of a pipeline run.  All agent telemetry is
ephemeral (Redis pub/sub); the ORM model stores only the essential lifecycle
fields that must survive a process restart.
"""
from __future__ import annotations

import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.core.database import Base


class Task(Base):
    """One row per /generate-video invocation."""

    __tablename__ = "tasks"

    # ── Identity ──────────────────────────────────────────────────────────────
    id: Mapped[str] = mapped_column(String(26), primary_key=True)  # ULID
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    # ── Request snapshot (stored as JSON text) ─────────────────────────────────
    request_json: Mapped[str] = mapped_column(Text, nullable=False)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="queued",
        server_default="queued",
    )
    stage: Mapped[str | None] = mapped_column(String(30), nullable=True)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Results ───────────────────────────────────────────────────────────────
    video_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    thumbnail_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    script_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Error ─────────────────────────────────────────────────────────────────
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Webhook ───────────────────────────────────────────────────────────────
    webhook_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    # ── Ownership ─────────────────────────────────────────────────────────────
    # Nullable so legacy tasks (created before auth was added) keep working.
    user_id: Mapped[str | None] = mapped_column(String(26), nullable=True, index=True)

    __table_args__ = (
        Index("ix_tasks_status_created", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Task id={self.id} status={self.status}>"


class User(Base):
    """Application user — used for frontend login and API access control."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(26), primary_key=True)  # ULID
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(20), nullable=False, default="user", server_default="user"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Max video generations per day. NULL = unlimited (admins default to NULL).
    daily_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username} role={self.role}>"
