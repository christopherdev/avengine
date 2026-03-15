"""
Celery pipeline dispatch task.

This is the entry-point called by the FastAPI POST /generate-video endpoint.
It:
  1. Marks the DB task as `running`.
  2. Builds the initial GraphState.
  3. Invokes the LangGraph pipeline synchronously (LangGraph is sync-native
     when run from a Celery worker).
  4. On success — persists artefact URLs, marks the task `completed`, fires webhook.
  5. On failure — marks the task `failed`, publishes an error event.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import structlog
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from src.core.celery_app import celery_app
from src.schemas.api import AgentStage, TaskStatus

logger = structlog.get_logger(__name__)


@celery_app.task(
    bind=True,
    name="src.pipeline.tasks.run_pipeline",
    max_retries=0,          # graph handles retries internally
    acks_late=True,
    reject_on_worker_lost=True,
    track_started=True,
)
def run_pipeline(self: Task, *, task_id: str, request_data: dict[str, Any]) -> dict[str, Any]:
    """
    Main pipeline orchestrator task.

    Runs entirely synchronously — LangGraph's MemorySaver checkpointer
    persists state to in-memory storage (swap for RedisCheckpointer in
    high-availability deployments).
    """
    log = logger.bind(task_id=task_id)
    log.info("pipeline.task_started")

    # ── Mark DB task as running ───────────────────────────────────────────────
    _update_task_status(task_id, TaskStatus.running, stage=AgentStage.ideation, progress=0.0)

    try:
        from src.pipeline.graph import get_graph
        from src.pipeline.state import initial_state

        graph = get_graph()

        # Strip internal retry metadata before building request state
        resume_task_id: str | None = request_data.pop("_resume_task_id", None)

        state = initial_state(task_id=task_id, request=request_data)

        # Restore checkpoint from the original task so completed stages are skipped
        if resume_task_id:
            from src.pipeline.checkpoint import load_checkpoint

            checkpoint = load_checkpoint(resume_task_id)
            if checkpoint:
                state["completed_nodes"] = checkpoint.get("completed_nodes", [])
                if checkpoint.get("brief") is not None:
                    state["brief"] = checkpoint["brief"]
                if checkpoint.get("script") is not None:
                    state["script"] = checkpoint["script"]
                if checkpoint.get("sourced_clips"):
                    state["sourced_clips"] = checkpoint["sourced_clips"]
                if checkpoint.get("matched_clips"):
                    state["matched_clips"] = checkpoint["matched_clips"]
                if checkpoint.get("timeline") is not None:
                    state["timeline"] = checkpoint["timeline"]
                log.info(
                    "pipeline.checkpoint_restored",
                    resume_from=resume_task_id,
                    completed_nodes=state["completed_nodes"],
                )
            else:
                log.warning("pipeline.checkpoint_missing", resume_from=resume_task_id)

        config = {"configurable": {"thread_id": task_id}}
        final_state: dict[str, Any] = graph.invoke(state, config=config)

        # ── Check for pipeline-level errors ───────────────────────────────────
        errors: list[dict] = final_state.get("errors", [])
        if errors and final_state.get("current_node") == "failed":
            msg = errors[-1].get("message", "Unknown pipeline error")
            _update_task_status(task_id, TaskStatus.failed, error=msg)
            log.error("pipeline.failed", error=msg)
            return {"task_id": task_id, "status": "failed", "error": msg}

        # ── Persist results ───────────────────────────────────────────────────
        video_url     = final_state.get("output_video_url")
        thumbnail_url = final_state.get("thumbnail_url")
        script_dict   = final_state.get("script") or {}
        timeline_dict = final_state.get("timeline") or {}

        _update_task_status(
            task_id,
            TaskStatus.completed,
            stage=AgentStage.done,
            progress=100.0,
            video_url=video_url,
            thumbnail_url=thumbnail_url,
            script_text=script_dict.get("raw_text"),
            duration_seconds=timeline_dict.get("total_duration"),
        )

        log.info("pipeline.completed", video_url=video_url)

        # ── Fire webhook (non-blocking) ───────────────────────────────────────
        webhook_url = _get_webhook_url(task_id)
        if webhook_url:
            _fire_webhook.delay(
                webhook_url=webhook_url,
                task_id=task_id,
                video_url=video_url,
                thumbnail_url=thumbnail_url,
            )

        return {"task_id": task_id, "status": "completed", "video_url": video_url}

    except SoftTimeLimitExceeded:
        msg = "Pipeline exceeded time limit and was terminated."
        _update_task_status(task_id, TaskStatus.failed, error=msg)
        log.error("pipeline.time_limit_exceeded")
        return {"task_id": task_id, "status": "failed", "error": msg}

    except Exception as exc:  # noqa: BLE001
        # Include detail (e.g. FFmpeg stderr) in the stored error if available
        detail = getattr(exc, "detail", None)
        msg = f"{exc}{(' — ' + detail) if detail else ''}"
        _update_task_status(task_id, TaskStatus.failed, error=msg)
        log.exception("pipeline.unhandled_exception")
        return {"task_id": task_id, "status": "failed", "error": msg}


# ── Webhook delivery ──────────────────────────────────────────────────────────

@celery_app.task(
    name="src.pipeline.tasks.fire_webhook",
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(Exception,),
)
def _fire_webhook(
    *,
    webhook_url: str,
    task_id: str,
    video_url: str | None,
    thumbnail_url: str | None,
) -> None:
    import httpx

    payload = {
        "task_id": task_id,
        "status": "completed",
        "video_url": video_url,
        "thumbnail_url": thumbnail_url,
    }
    response = httpx.post(webhook_url, json=payload, timeout=10)
    response.raise_for_status()
    logger.info("webhook.delivered", task_id=task_id, url=webhook_url)


# ── DB helpers (sync wrappers around async SQLAlchemy) ────────────────────────

def _run_sync(coro: Any) -> Any:
    """Run a coroutine from synchronous Celery worker context."""
    return asyncio.run(coro)


def _update_task_status(
    task_id: str,
    status: TaskStatus,
    *,
    stage: AgentStage | None = None,
    progress: float | None = None,
    error: str | None = None,
    video_url: str | None = None,
    thumbnail_url: str | None = None,
    script_text: str | None = None,
    duration_seconds: float | None = None,
) -> None:
    async def _update() -> None:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
        from sqlalchemy.pool import NullPool

        from src.core.config import get_settings
        from src.core.models import Task

        s = get_settings()
        # NullPool avoids binding connections to a specific event loop,
        # which would break across multiple asyncio.run() calls in Celery.
        engine = create_async_engine(s.database_url, poolclass=NullPool)
        _sf = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with _sf() as session:
            task = await session.get(Task, task_id)
            if task is None:
                return

            task.status = status.value
            task.updated_at = datetime.now(tz=timezone.utc)

            if stage is not None:
                task.stage = stage.value
            if progress is not None:
                task.progress = progress
            if error is not None:
                task.error_message = error
            if video_url is not None:
                task.video_url = video_url
            if thumbnail_url is not None:
                task.thumbnail_url = thumbnail_url
            if script_text is not None:
                task.script_text = script_text
            if duration_seconds is not None:
                task.duration_seconds = duration_seconds
            if status in (TaskStatus.completed, TaskStatus.failed):
                task.completed_at = datetime.now(tz=timezone.utc)

            await session.commit()

    _run_sync(_update())


def _get_webhook_url(task_id: str) -> str | None:
    async def _fetch() -> str | None:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
        from sqlalchemy.pool import NullPool

        from src.core.config import get_settings
        from src.core.models import Task

        s = get_settings()
        engine = create_async_engine(s.database_url, poolclass=NullPool)
        _sf = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with _sf() as session:
            task = await session.get(Task, task_id)
            return task.webhook_url if task else None

    return _run_sync(_fetch())
