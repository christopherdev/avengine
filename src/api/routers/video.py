"""
Video generation REST router.

  POST   /generate-video              → 202 Accepted  (enqueue pipeline)
  GET    /tasks                       → paginated task list
  GET    /tasks/{task_id}             → task status (polling)
  GET    /tasks/{task_id}/result      → final artefact URLs
  POST   /tasks/{task_id}/retry       → re-queue failed task
  DELETE /tasks/{task_id}             → cancel / soft-delete

Task visibility:
  - Admin users see all tasks.
  - Regular authenticated users see only their own tasks.
  - Unauthenticated requests (dev mode) see all tasks (no user_id filter).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Annotated

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from fastapi.responses import ORJSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from src.api.dependencies.auth import get_optional_current_user
from src.core.database import get_session
from src.core.models import Task, User
from src.schemas.api import (
    AcceptedResponse,
    AgentStage,
    GenerateVideoRequest,
    TaskListResponse,
    TaskResult,
    TaskStatus,
    TaskStatusResponse,
)

router = APIRouter(prefix="/generate-video", tags=["Video Generation"])
tasks_router = APIRouter(prefix="/tasks", tags=["Tasks"])
logger = structlog.get_logger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

SessionDep = Annotated[AsyncSession, Depends(get_session)]
OptionalUserDep = Annotated[User | None, Depends(get_optional_current_user)]


def _task_to_status(task: Task) -> TaskStatusResponse:
    try:
        req = json.loads(task.request_json)
        topic = req.get("topic")
        style = req.get("style")
    except Exception:
        topic = None
        style = None
    return TaskStatusResponse(
        task_id=task.id,
        topic=topic,
        style=style,
        status=TaskStatus(task.status),
        stage=AgentStage(task.stage) if task.stage else None,
        progress=task.progress,
        created_at=task.created_at,
        updated_at=task.updated_at,
        error=task.error_message,
    )


async def _get_task_or_404(
    task_id: str, session: AsyncSession, current_user: User | None
) -> Task:
    task = await session.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    _check_ownership(task, current_user)
    return task


def _check_ownership(task: Task, current_user: User | None) -> None:
    """Raise 404 if a non-admin user tries to access another user's task."""
    if current_user is None:
        return  # dev mode / no auth — allow all
    if current_user.role == "admin":
        return  # admins see everything
    if task.user_id is not None and task.user_id != current_user.id:
        # Return 404 (not 403) to avoid leaking task existence
        raise HTTPException(status_code=404, detail=f"Task '{task.id}' not found.")


async def _check_daily_limit(user: User, session: AsyncSession) -> None:
    """Raise 429 if the user has hit their daily generation limit."""
    if user.daily_limit is None:
        return  # unlimited

    today_start = datetime.now(tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    count_stmt = (
        select(func.count())
        .select_from(Task)
        .where(Task.user_id == user.id)
        .where(Task.created_at >= today_start)
    )
    count: int = (await session.execute(count_stmt)).scalar_one()

    if count >= user.daily_limit:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily generation limit of {user.daily_limit} reached. "
                "Try again tomorrow."
            ),
        )


# ── POST /generate-video ──────────────────────────────────────────────────────

@router.post(
    "",
    status_code=202,
    response_model=AcceptedResponse,
    response_class=ORJSONResponse,
    summary="Enqueue a video generation pipeline",
    responses={
        202: {"description": "Pipeline accepted and queued."},
        422: {"description": "Invalid request body."},
        429: {"description": "Rate limit exceeded."},
    },
)
async def generate_video(
    body: GenerateVideoRequest,
    request: Request,
    session: SessionDep,
    background_tasks: BackgroundTasks,
    current_user: OptionalUserDep,
) -> AcceptedResponse:
    """
    Accepts a video brief, persists a Task row, and dispatches the
    LangGraph pipeline to the Celery `pipeline` queue.

    Returns **202 Accepted** immediately with:
    - `task_id`   — ULID for polling / WebSocket subscription
    - `status_url` — polling endpoint
    - `ws_url`    — WebSocket for real-time events
    """
    if current_user is not None:
        await _check_daily_limit(current_user, session)

    task_id = str(ULID())

    # ── Persist task ──────────────────────────────────────────────────────────
    task = Task(
        id=task_id,
        request_json=body.model_dump_json(),
        status=TaskStatus.queued.value,
        webhook_url=str(body.webhook_url) if body.webhook_url else None,
        user_id=current_user.id if current_user else None,
    )
    session.add(task)
    await session.flush()  # write before Celery dispatch

    # ── Dispatch to Celery ────────────────────────────────────────────────────
    from src.pipeline.tasks import run_pipeline  # avoid circular at module load

    celery_result = run_pipeline.apply_async(
        kwargs={"task_id": task_id, "request_data": body.model_dump(mode="json")},
        task_id=task_id,        # Celery task ID == AVEngine task ID for simplicity
    )
    task.celery_task_id = celery_result.id

    logger.info("video.enqueued", task_id=task_id, topic=body.topic[:80])

    # ── Build response URLs ───────────────────────────────────────────────────
    base = str(request.base_url).rstrip("/")
    prefix = request.app.state.api_prefix if hasattr(request.app.state, "api_prefix") else "/api/v1"

    return AcceptedResponse(
        task_id=task_id,
        status_url=f"{base}{prefix}/tasks/{task_id}",
        ws_url=f"{base.replace('http', 'ws')}{prefix}/ws/tasks/{task_id}",
    )


# ── GET /tasks ────────────────────────────────────────────────────────────────

@tasks_router.get(
    "",
    response_model=TaskListResponse,
    response_class=ORJSONResponse,
    summary="List tasks (paginated)",
)
async def list_tasks(
    session: SessionDep,
    current_user: OptionalUserDep,
    status: TaskStatus | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> TaskListResponse:
    stmt = select(Task).order_by(Task.created_at.desc())
    if status:
        stmt = stmt.where(Task.status == status.value)

    # Non-admin authenticated users see only their own tasks.
    # Unauthenticated requests see only tasks submitted without authentication.
    if current_user is not None and current_user.role != "admin":
        stmt = stmt.where(Task.user_id == current_user.id)
    elif current_user is None:
        stmt = stmt.where(Task.user_id.is_(None))

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = (await session.execute(count_stmt)).scalar_one()

    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    rows = (await session.execute(stmt)).scalars().all()

    return TaskListResponse(
        items=[_task_to_status(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


# ── GET /tasks/{task_id} ──────────────────────────────────────────────────────

@tasks_router.get(
    "/{task_id}",
    response_model=TaskStatusResponse,
    response_class=ORJSONResponse,
    summary="Poll task status",
)
async def get_task_status(
    task_id: str, session: SessionDep, current_user: OptionalUserDep
) -> TaskStatusResponse:
    task = await _get_task_or_404(task_id, session, current_user)
    return _task_to_status(task)


# ── GET /tasks/{task_id}/result ───────────────────────────────────────────────

@tasks_router.get(
    "/{task_id}/result",
    response_model=TaskResult,
    response_class=ORJSONResponse,
    summary="Retrieve final artefacts",
)
async def get_task_result(
    task_id: str, session: SessionDep, current_user: OptionalUserDep
) -> TaskResult:
    task = await _get_task_or_404(task_id, session, current_user)

    if task.status not in (TaskStatus.completed.value, TaskStatus.failed.value):
        raise HTTPException(
            status_code=409,
            detail=f"Task is in '{task.status}' state — result not yet available.",
        )

    return TaskResult(
        task_id=task.id,
        status=TaskStatus(task.status),
        video_url=task.video_url,
        thumbnail_url=task.thumbnail_url,
        script=task.script_text,
        duration_seconds=task.duration_seconds,
        completed_at=task.completed_at,
        error=task.error_message,
    )


# ── POST /tasks/{task_id}/retry ───────────────────────────────────────────────

@tasks_router.post(
    "/{task_id}/retry",
    status_code=202,
    response_model=AcceptedResponse,
    response_class=ORJSONResponse,
    summary="Re-queue a failed task with the same parameters",
)
async def retry_task(
    task_id: str,
    request: Request,
    session: SessionDep,
    current_user: OptionalUserDep,
) -> AcceptedResponse:
    task = await _get_task_or_404(task_id, session, current_user)

    if task.status not in (TaskStatus.failed.value, TaskStatus.cancelled.value):
        raise HTTPException(
            status_code=409,
            detail=f"Only failed or cancelled tasks can be retried (current: '{task.status}').",
        )

    if current_user is not None:
        await _check_daily_limit(current_user, session)

    try:
        request_data = json.loads(task.request_json)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Original request data is corrupt.") from exc

    # Tell the worker to restore the checkpoint from this task's output directory
    request_data["_resume_task_id"] = task_id

    new_task_id = str(ULID())
    new_task = Task(
        id=new_task_id,
        request_json=task.request_json,  # store clean original (without _resume_task_id)
        status=TaskStatus.queued.value,
        webhook_url=task.webhook_url,
        user_id=task.user_id,  # preserve ownership
    )
    session.add(new_task)
    await session.flush()

    from src.pipeline.tasks import run_pipeline

    celery_result = run_pipeline.apply_async(
        kwargs={"task_id": new_task_id, "request_data": request_data},
        task_id=new_task_id,
    )
    new_task.celery_task_id = celery_result.id

    logger.info("task.retried", original_task_id=task_id, new_task_id=new_task_id)

    base = str(request.base_url).rstrip("/")
    prefix = request.app.state.api_prefix if hasattr(request.app.state, "api_prefix") else "/api/v1"

    return AcceptedResponse(
        task_id=new_task_id,
        status_url=f"{base}{prefix}/tasks/{new_task_id}",
        ws_url=f"{base.replace('http', 'ws')}{prefix}/ws/tasks/{new_task_id}",
    )


# ── DELETE /tasks/{task_id} ───────────────────────────────────────────────────

@tasks_router.delete(
    "/{task_id}",
    status_code=204,
    summary="Cancel or delete a task",
)
async def cancel_task(
    task_id: str, session: SessionDep, current_user: OptionalUserDep
) -> None:
    task = await _get_task_or_404(task_id, session, current_user)

    if task.status in (TaskStatus.completed.value, TaskStatus.failed.value):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel a task in terminal state '{task.status}'.",
        )

    # Revoke from Celery (best-effort — worker may have already picked it up)
    if task.celery_task_id:
        from src.core.celery_app import celery_app

        celery_app.control.revoke(task.celery_task_id, terminate=True, signal="SIGTERM")

    task.status = TaskStatus.cancelled.value
    task.updated_at = datetime.now(tz=timezone.utc)
    logger.info("task.cancelled", task_id=task_id)
