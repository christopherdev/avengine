"""
LangGraph-compatible state definition.

LangGraph requires a TypedDict (or dataclass) as its graph state.
We define `GraphState` here as the TypedDict, then provide helpers to
marshal to/from the richer `PipelineState` Pydantic model.

The TypedDict uses `Annotated` reducer fields so LangGraph can merge
partial node outputs correctly (e.g. appending errors, clips).
"""
from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph import add_messages
from typing_extensions import TypedDict


def _append(existing: list, new: list) -> list:
    """Reducer: append new items to the existing list."""
    return existing + new


def _replace(existing: Any, new: Any) -> Any:  # noqa: ANN401
    """Reducer: replace the existing value with the new one (default behaviour)."""
    return new if new is not None else existing


class GraphState(TypedDict, total=False):
    """
    LangGraph state for the AVEngine pipeline.

    Each key maps to a pipeline concern.  Reducer annotations tell
    LangGraph how to merge concurrent or sequential node outputs.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    task_id: str
    request: dict[str, Any]

    # ── Node outputs ──────────────────────────────────────────────────────────
    brief: dict[str, Any] | None                              # IdeationBrief
    script: dict[str, Any] | None                             # VideoScript
    sourced_clips: Annotated[list[dict[str, Any]], _append]   # SourcedClip[]
    matched_clips: Annotated[list[dict[str, Any]], _append]   # MatchedClip[]
    timeline: dict[str, Any] | None                           # VideoTimeline

    # ── Rendered output ────────────────────────────────────────────────────────
    output_video_path: str | None
    output_video_url: str | None
    thumbnail_url: str | None

    # ── Control flow ──────────────────────────────────────────────────────────
    current_node: str
    errors: Annotated[list[dict[str, Any]], _append]
    retry_count: int
    is_cancelled: bool

    # ── Checkpoint tracking ───────────────────────────────────────────────────
    completed_nodes: Annotated[list[str], _append]  # nodes that have already run


def initial_state(task_id: str, request: dict[str, Any]) -> GraphState:
    """Return a fully-initialised GraphState for a new pipeline run."""
    return GraphState(
        task_id=task_id,
        request=request,
        brief=None,
        script=None,
        sourced_clips=[],
        matched_clips=[],
        timeline=None,
        output_video_path=None,
        output_video_url=None,
        thumbnail_url=None,
        current_node="ideation",
        errors=[],
        retry_count=0,
        is_cancelled=False,
        completed_nodes=[],
    )
