"""
File-based pipeline checkpoint.

After each node succeeds, its outputs are written to:
  {video_output_dir}/{task_id}/checkpoint.json

On retry, the new task loads the checkpoint from the original task and
pre-populates its GraphState so completed nodes are skipped automatically.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _checkpoint_path(task_id: str) -> pathlib.Path:
    from src.core.config import get_settings
    return pathlib.Path(get_settings().video_output_dir) / task_id / "checkpoint.json"


def save_checkpoint(task_id: str, data: dict[str, Any]) -> None:
    """Persist checkpoint data to disk alongside the task output. Non-fatal."""
    try:
        path = _checkpoint_path(task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        logger.debug("checkpoint.saved", task_id=task_id,
                     completed=data.get("completed_nodes", []))
    except Exception as exc:
        logger.warning("checkpoint.save_failed", task_id=task_id, error=str(exc))


def load_checkpoint(task_id: str) -> dict[str, Any] | None:
    """Load checkpoint data from disk. Returns None if not found or corrupt."""
    try:
        path = _checkpoint_path(task_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        logger.info("checkpoint.loaded", task_id=task_id,
                    completed=data.get("completed_nodes", []))
        return data
    except Exception as exc:
        logger.warning("checkpoint.load_failed", task_id=task_id, error=str(exc))
        return None
