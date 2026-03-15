"""
WebSocket router — real-time agent state broadcasting.

  WS /ws/tasks/{task_id}

Protocol:
  1. Client connects.
  2. Server immediately sends the current task status snapshot (from DB).
  3. Server subscribes to Redis channel `task:<task_id>` and fans out every
     event as a JSON text frame.
  4. On terminal event (type == "done" | "error"), server sends the event
     and closes the connection with code 1000.
  5. If the client disconnects early, the Redis subscription is torn down.

Message envelope — matches TaskEvent schema:
  {
    "task_id": "...",
    "type":    "scripting",      // AgentStage
    "message": "Generating script...",
    "progress": 35.0,
    "data":    {},
    "timestamp": "2025-01-01T00:00:00Z"
  }
"""
from __future__ import annotations

import json

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.websockets import WebSocketState

from src.core.database import get_session
from src.core.redis_bus import get_redis_bus
from src.schemas.api import AgentStage, TaskStatus

router = APIRouter(prefix="/ws", tags=["WebSocket"])
logger = structlog.get_logger(__name__)

_TERMINAL = {AgentStage.done.value, AgentStage.error.value}


async def _authenticate_ws(websocket: WebSocket) -> str | None:
    """
    Validate ?token=<jwt> query parameter.

    Returns the user_id on success, or None if absent / invalid.
    Does NOT close the socket — caller handles rejection after accept().
    """
    from fastapi.security import HTTPAuthorizationCredentials

    from src.api.dependencies.auth import _decode_bearer

    token: str | None = websocket.query_params.get("token")
    if not token:
        return None

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    payload = await _decode_bearer(creds)
    if payload is None:
        return None

    return payload.get("sub")


@router.websocket("/tasks/{task_id}")
async def task_events(websocket: WebSocket, task_id: str) -> None:
    """
    Real-time event stream for a pipeline run.

    Authentication: pass ``?token=<jwt>`` as a query parameter.
    In development with ALLOW_ANONYMOUS_DEV=true, token-less connections are
    accepted.  Otherwise a missing or invalid token results in an immediate
    close (4401) after the handshake.

    The connection lifecycle:
      connect → auth check → snapshot → subscribe loop → terminal event → close
    """
    from src.core.config import get_settings as _get_settings

    user_id = await _authenticate_ws(websocket)
    _settings = _get_settings()

    # Reject unauthenticated connections unless anonymous dev mode is on
    if user_id is None and not _settings.allow_anonymous_dev:
        await websocket.accept()
        await websocket.close(code=4401, reason="Authentication required.")
        return

    await websocket.accept()
    log = logger.bind(task_id=task_id, client=websocket.client)
    log.info("ws.connected", user_id=user_id)

    try:
        # ── 1. Send current status snapshot from DB ───────────────────────────
        async for session in get_session():
            snapshot = await _build_snapshot(task_id, session)
            break  # only need one iteration

        await _send(websocket, snapshot)

        # If the task is already terminal, close immediately
        if snapshot.get("type") in _TERMINAL:
            await websocket.close(code=1000)
            return

        # ── 2. Subscribe to Redis pub/sub ─────────────────────────────────────
        bus = get_redis_bus(websocket.app.state)  # type: ignore[attr-defined]

        async for event in bus.subscribe(task_id):
            if websocket.client_state != WebSocketState.CONNECTED:
                break

            await _send(websocket, event)

            if event.get("type") in _TERMINAL:
                await websocket.close(code=1000)
                break

    except WebSocketDisconnect:
        log.info("ws.client_disconnected")
    except Exception as exc:  # noqa: BLE001
        log.error("ws.error", error=str(exc))
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011)  # Internal error
    finally:
        log.info("ws.closed")


# ── Connection Manager (broadcast helper used by pipeline nodes) ───────────────

class ConnectionManager:
    """
    In-process registry of active WebSocket connections.

    Used by the pipeline when running in the *same process* as the API
    (development mode).  In production, the Celery worker publishes to Redis
    and the API's Redis subscriber fans out to WebSockets — this class is
    not needed for the Redis path.
    """

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, task_id: str, ws: WebSocket) -> None:
        self._connections.setdefault(task_id, []).append(ws)

    def disconnect(self, task_id: str, ws: WebSocket) -> None:
        conns = self._connections.get(task_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, task_id: str, event: dict) -> None:
        dead: list[WebSocket] = []
        for ws in self._connections.get(task_id, []):
            try:
                await ws.send_text(json.dumps(event, default=str))
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.disconnect(task_id, ws)


manager = ConnectionManager()


# ── Internal helpers ──────────────────────────────────────────────────────────

async def _build_snapshot(task_id: str, session: AsyncSession) -> dict:
    """
    Read current task row from DB and format as a TaskEvent-compatible dict.
    Returns an error event if the task doesn't exist.
    """
    from src.core.models import Task

    task = await session.get(Task, task_id)
    if task is None:
        return {
            "task_id": task_id,
            "type": AgentStage.error.value,
            "message": f"Task '{task_id}' not found.",
            "progress": 0.0,
            "data": {},
        }

    # Map DB status → AgentStage for the snapshot
    if task.status == TaskStatus.completed.value:
        stage = AgentStage.done.value
    elif task.status == TaskStatus.failed.value:
        stage = AgentStage.error.value
    else:
        stage = task.stage or AgentStage.ideation.value

    return {
        "task_id": task_id,
        "type": stage,
        "message": f"Task is {task.status}.",
        "progress": task.progress,
        "data": {"status": task.status},
    }


async def _send(websocket: WebSocket, event: dict) -> None:
    await websocket.send_text(json.dumps(event, default=str))
