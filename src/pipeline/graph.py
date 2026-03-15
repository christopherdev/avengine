"""
LangGraph state machine — the macro-workflow orchestrator.

Nodes (in execution order):
  ideation   → Distils the brief into a structured IdeationBrief
  scripting  → CrewAI crew: Researcher → Writer → Editor → raw script text
  extracting → Pydantic AI: raw text → validated VideoScript JSON
  sourcing   → Async crawler: yt-dlp + Playwright per B-roll cue
  matching   → TwelveLabs embeddings → Qdrant query → MatchedClip[]
  rendering  → MoviePy v2 + FFmpeg → final MP4

Edge conditions:
  - Any node can transition to `handle_error` on exception.
  - `handle_error` checks if the error is recoverable and either retries
    (up to MAX_RETRIES) or transitions to END with a failure status.
  - `check_cancelled` is evaluated after every node to honour cancellations
    submitted via DELETE /tasks/{id}.
"""
from __future__ import annotations

import traceback
from typing import Any

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.pipeline.state import GraphState
from src.schemas.api import AgentStage

logger = structlog.get_logger(__name__)

MAX_RETRIES = 2

# ── Progress constants per node (0–100) ───────────────────────────────────────
_PROGRESS = {
    "ideation":   5.0,
    "scripting":  20.0,
    "extracting": 30.0,
    "sourcing":   55.0,
    "matching":   70.0,
    "rendering":  95.0,
    "done":       100.0,
}


# ── Node: ideation ─────────────────────────────────────────────────────────────

def node_ideation(state: GraphState) -> dict[str, Any]:
    """
    Transforms the raw topic + style from the request into a structured
    IdeationBrief that guides the scripting crew.
    """
    if "ideation" in state.get("completed_nodes", []):
        logger.info("node.ideation.skip", task_id=state.get("task_id"))
        return {"current_node": "scripting"}

    from src.pipeline.publisher import publish_event

    task_id: str = state["task_id"]
    request: dict = state["request"]

    publish_event(
        task_id, AgentStage.ideation,
        "Generating video brief...", _PROGRESS["ideation"],
    )
    logger.info("node.ideation.start", task_id=task_id)

    from langchain_openai import ChatOpenAI

    from src.core.config import get_settings

    settings = get_settings()
    llm = ChatOpenAI(model=settings.openai_model, temperature=0.7)

    topic = request.get("topic", "")
    style = request.get("style", "explainer")
    duration = request.get("duration_seconds", 60)
    audience = request.get("target_audience", "general")

    is_reels = style == "reels"
    wpm = 130  # average speaking rate
    estimated_words = int((duration / 60) * wpm)

    reels_rules = """
REELS FORMAT RULES (9:16 vertical, max 60s):
- Hook must grab attention in the FIRST 3 words — use a bold claim or question
- key_points: exactly 3 items, each punchy and visual
- search_queries: visually striking, fast-paced footage (time-lapses, close-ups, action)
- tone must be: inspirational or conversational (never authoritative)
- estimated_word_count: keep under 120 words total
""" if is_reels else ""

    prompt = f"""You are a video strategist. Given the topic below, create a structured creative brief.

Topic: {topic}
Style: {style}
Target Audience: {audience}
Target Duration: {duration} seconds (~{estimated_words} words of narration)
{reels_rules}
Return ONLY a JSON object with these exact keys:
{{
  "title": "...",
  "hook": "One compelling opening sentence (max 150 chars)",
  "key_points": ["point1", "point2", "point3"],
  "tone": "conversational",
  "target_audience": "{audience}",
  "estimated_word_count": {estimated_words},
  "search_queries": ["query1", "query2", "query3"]
}}

Rules:
- key_points: 3-7 items (3 if reels)
- search_queries: 3-8 YouTube/stock-footage search queries for B-roll
- tone: one of authoritative|conversational|inspirational|humorous|neutral
"""

    import json

    from src.schemas.pipeline import IdeationBrief

    response = llm.invoke(prompt)
    raw = response.content

    # Strip markdown fences if present
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    brief_dict = json.loads(raw.strip())
    # Inject request-level fields the LLM doesn't return
    brief_dict.setdefault("style", style)
    brief_dict.setdefault("duration_seconds", duration)
    brief = IdeationBrief(**brief_dict)

    logger.info("node.ideation.done", task_id=task_id, title=brief.title)

    result = {"brief": brief.model_dump(), "current_node": "scripting", "completed_nodes": ["ideation"]}
    _checkpoint_node(state, "ideation", result)
    return result


# ── Node: scripting ────────────────────────────────────────────────────────────

def node_scripting(state: GraphState) -> dict[str, Any]:
    """
    Runs the CrewAI Script Generation crew.
    Returns raw script text that the extracting node will parse.
    """
    if "scripting" in state.get("completed_nodes", []):
        logger.info("node.scripting.skip", task_id=state.get("task_id"))
        return {"current_node": "extracting"}

    from src.pipeline.publisher import publish_event

    task_id: str = state["task_id"]
    brief_dict: dict = state["brief"]  # type: ignore[assignment]

    publish_event(
        task_id, AgentStage.scripting,
        "Crew is researching and writing the script...", _PROGRESS["scripting"],
    )
    logger.info("node.scripting.start", task_id=task_id)

    from src.agents.script.crew import ScriptCrew

    crew = ScriptCrew()
    raw_script_text: str = crew.run(brief=brief_dict)

    logger.info(
        "node.scripting.done",
        task_id=task_id,
        chars=len(raw_script_text),
    )

    result = {
        "script": {"raw_text": raw_script_text, "title": brief_dict.get("title", "")},
        "current_node": "extracting",
        "completed_nodes": ["scripting"],
    }
    _checkpoint_node(state, "scripting", result)
    return result


# ── Node: extracting ───────────────────────────────────────────────────────────

def node_extracting(state: GraphState) -> dict[str, Any]:
    """
    Pydantic AI structured extraction node.
    Converts the raw script text produced by CrewAI into a validated
    VideoScript object.
    """
    if "extracting" in state.get("completed_nodes", []):
        logger.info("node.extracting.skip", task_id=state.get("task_id"))
        return {"current_node": "sourcing"}

    from src.pipeline.publisher import publish_event

    task_id: str = state["task_id"]
    script_partial: dict = state["script"]  # type: ignore[assignment]

    publish_event(
        task_id, AgentStage.scripting,
        "Extracting structured timeline from script...", _PROGRESS["extracting"],
    )
    logger.info("node.extracting.start", task_id=task_id)

    from src.agents.script.extractor import extract_video_script

    video_script = extract_video_script(
        raw_text=script_partial["raw_text"],
        title=script_partial.get("title", ""),
    )

    logger.info(
        "node.extracting.done",
        task_id=task_id,
        scenes=len(video_script.scenes),
    )

    result = {"script": video_script.model_dump(), "current_node": "sourcing", "completed_nodes": ["extracting"]}
    _checkpoint_node(state, "extracting", result)
    return result


# ── Node: sourcing ─────────────────────────────────────────────────────────────

def node_sourcing(state: GraphState) -> dict[str, Any]:
    """
    Dispatches B-roll search queries to the async crawler.
    Returns a flat list of SourcedClip objects.
    """
    if "sourcing" in state.get("completed_nodes", []):
        logger.info("node.sourcing.skip", task_id=state.get("task_id"),
                    clips=len(state.get("sourced_clips", [])))
        return {"current_node": "matching"}

    from src.pipeline.publisher import publish_event

    task_id: str = state["task_id"]
    script_dict: dict = state["script"]  # type: ignore[assignment]
    request: dict = state["request"]

    publish_event(
        task_id, AgentStage.sourcing,
        "Sourcing B-roll footage...", _PROGRESS["sourcing"],
    )
    logger.info("node.sourcing.start", task_id=task_id)

    from src.agents.sourcing.worker import SourcingWorker

    worker = SourcingWorker()
    sourced = worker.run_sync(
        script_dict=script_dict,
        seed_urls=request.get("seed_urls", []),
        task_id=task_id,
        topic=request.get("topic", ""),
    )

    logger.info("node.sourcing.done", task_id=task_id, clips=len(sourced))

    result = {"sourced_clips": [c.model_dump() for c in sourced], "current_node": "matching", "completed_nodes": ["sourcing"]}
    _checkpoint_node(state, "sourcing", result)
    return result


# ── Node: matching ─────────────────────────────────────────────────────────────

def node_matching(state: GraphState) -> dict[str, Any]:
    """
    Uses TwelveLabs + Qdrant to semantically match each B-roll cue to the
    best available clip segment.
    """
    if "matching" in state.get("completed_nodes", []):
        logger.info("node.matching.skip", task_id=state.get("task_id"),
                    clips=len(state.get("matched_clips", [])))
        return {"current_node": "rendering"}

    from src.pipeline.publisher import publish_event

    task_id: str = state["task_id"]
    script_dict: dict = state["script"]  # type: ignore[assignment]
    sourced: list[dict] = state.get("sourced_clips", [])

    publish_event(
        task_id, AgentStage.matching,
        "Matching script sentences to video clips...", _PROGRESS["matching"],
    )
    logger.info("node.matching.start", task_id=task_id, sourced_count=len(sourced))

    from src.agents.matching.matcher import SemanticMatcher

    matcher = SemanticMatcher()
    matched, timeline = matcher.run_sync(
        task_id=task_id,
        script_dict=script_dict,
        sourced_clips=sourced,
        request=state["request"],
    )

    logger.info("node.matching.done", task_id=task_id, matched=len(matched))

    result = {
        "matched_clips": [c.model_dump() for c in matched],
        "timeline": timeline.model_dump(),
        "current_node": "rendering",
        "completed_nodes": ["matching"],
    }
    _checkpoint_node(state, "matching", result)
    return result


# ── Node: rendering ────────────────────────────────────────────────────────────

def node_rendering(state: GraphState) -> dict[str, Any]:
    """
    Assembles the final video using MoviePy v2 + FFmpeg hybrid rendering.
    Uploads to S3 and returns the public URL.
    """
    from src.pipeline.publisher import publish_event

    task_id: str = state["task_id"]
    timeline_dict: dict = state["timeline"]  # type: ignore[assignment]

    publish_event(
        task_id, AgentStage.rendering,
        "Rendering final video...", _PROGRESS["rendering"],
    )
    logger.info("node.rendering.start", task_id=task_id)

    from src.agents.rendering.editor import VideoEditor

    editor = VideoEditor()
    video_path, thumbnail_path = editor.render_sync(
        task_id=task_id,
        timeline_dict=timeline_dict,
    )

    # Upload to S3
    from src.utils.storage import upload_to_s3

    video_url = upload_to_s3(video_path, s3_key=f"videos/{task_id}/output.mp4")
    thumbnail_url = upload_to_s3(thumbnail_path, s3_key=f"videos/{task_id}/thumbnail.jpg")

    publish_event(
        task_id, AgentStage.done,
        "Video ready.", _PROGRESS["done"],
        data={"video_url": video_url, "thumbnail_url": thumbnail_url},
    )

    logger.info("node.rendering.done", task_id=task_id, video_url=video_url)

    return {
        "output_video_path": video_path,
        "output_video_url": video_url,
        "thumbnail_url": thumbnail_url,
        "current_node": "done",
    }


# ── Node: handle_error ─────────────────────────────────────────────────────────

def node_handle_error(state: GraphState) -> dict[str, Any]:
    """
    Centralised error handler.  Receives the error injected by the
    node wrapper and decides whether to retry or terminate.
    """
    from src.pipeline.publisher import publish_event

    task_id: str = state["task_id"]
    errors: list[dict] = state.get("errors", [])
    retry_count: int = state.get("retry_count", 0)
    current_node: str = state.get("current_node", "unknown")

    last_error = errors[-1] if errors else {}
    is_recoverable = last_error.get("recoverable", False)

    logger.error(
        "node.error",
        task_id=task_id,
        node=current_node,
        error=last_error.get("message"),
        retry=retry_count,
    )

    if is_recoverable and retry_count < MAX_RETRIES:
        return {"retry_count": retry_count + 1, "current_node": current_node}

    # Terminal failure
    publish_event(
        task_id, AgentStage.error,
        f"Pipeline failed at {current_node}: {last_error.get('message', 'unknown error')}",
        progress=0.0,
    )
    return {"current_node": "failed"}


# ── Edge condition helpers ─────────────────────────────────────────────────────

def _route_after_node(state: GraphState) -> str:
    """Route to error handler if errors were appended, else continue."""
    if state.get("is_cancelled"):
        return "cancelled"
    errors: list = state.get("errors", [])
    if errors and errors[-1].get("node") == state.get("current_node"):
        return "handle_error"
    return state.get("current_node", "handle_error")


def _route_after_error(state: GraphState) -> str:
    current = state.get("current_node", "failed")
    if current == "failed":
        return END
    # Retry: go back to the failing node
    return current


# ── Checkpoint helper ──────────────────────────────────────────────────────────

def _checkpoint_node(state: GraphState, node_name: str, delta: dict[str, Any]) -> None:
    """
    Save the accumulated pipeline state to disk after a node succeeds.
    Merges the current state with the node's delta to capture all outputs so far.
    """
    from src.pipeline.checkpoint import save_checkpoint

    task_id = state.get("task_id", "")
    if not task_id:
        return

    save_checkpoint(task_id, {
        "completed_nodes": list(state.get("completed_nodes", [])) + [node_name],
        "brief":         delta.get("brief")     or state.get("brief"),
        "script":        delta.get("script")    or state.get("script"),
        "timeline":      delta.get("timeline")  or state.get("timeline"),
        # _append reducer fields: accumulate current state + any new items in delta
        "sourced_clips": state.get("sourced_clips", []) + delta.get("sourced_clips", []),
        "matched_clips": state.get("matched_clips", []) + delta.get("matched_clips", []),
    })


# ── Safe node wrapper ──────────────────────────────────────────────────────────

def _safe(node_fn: Any) -> Any:
    """
    Wrap a node function so unhandled exceptions are converted to error-state
    updates instead of crashing the graph.
    """
    def wrapper(state: GraphState) -> dict[str, Any]:
        try:
            return node_fn(state)
        except Exception as exc:  # noqa: BLE001
            node_name = node_fn.__name__.replace("node_", "")
            tb = traceback.format_exc()
            logger.error("node.exception", node=node_name, error=str(exc), tb=tb)
            return {
                "errors": [{"node": node_name, "message": str(exc), "recoverable": False}],
                "current_node": node_name,
            }

    wrapper.__name__ = node_fn.__name__
    return wrapper


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Compile and return the LangGraph state machine.

    The graph is stateless — each invocation receives a fresh GraphState
    dict.  The MemorySaver checkpointer enables mid-graph resumption after
    a Celery worker crash (the state is replayed from the last checkpoint).
    """
    builder = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("ideation",     _safe(node_ideation))
    builder.add_node("scripting",    _safe(node_scripting))
    builder.add_node("extracting",   _safe(node_extracting))
    builder.add_node("sourcing",     _safe(node_sourcing))
    builder.add_node("matching",     _safe(node_matching))
    builder.add_node("rendering",    _safe(node_rendering))
    builder.add_node("handle_error", node_handle_error)

    # ── Entry edge ────────────────────────────────────────────────────────────
    builder.add_edge(START, "ideation")

    # ── Sequential happy path with error branching ────────────────────────────
    for node, next_node in [
        ("ideation",   "scripting"),
        ("scripting",  "extracting"),
        ("extracting", "sourcing"),
        ("sourcing",   "matching"),
        ("matching",   "rendering"),
    ]:
        builder.add_conditional_edges(
            node,
            _route_after_node,
            {
                next_node:     next_node,
                "handle_error": "handle_error",
                "cancelled":    END,
            },
        )

    # ── Rendering → END ───────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "rendering",
        _route_after_node,
        {
            "done":         END,
            "handle_error": "handle_error",
            "cancelled":    END,
        },
    )

    # ── Error handler routing ─────────────────────────────────────────────────
    builder.add_conditional_edges(
        "handle_error",
        _route_after_error,
        {
            # Retry destinations
            "ideation":   "ideation",
            "scripting":  "scripting",
            "extracting": "extracting",
            "sourcing":   "sourcing",
            "matching":   "matching",
            "rendering":  "rendering",
            END:          END,
        },
    )

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Module-level compiled graph — built once per worker process
_graph: StateGraph | None = None


def get_graph() -> StateGraph:
    global _graph  # noqa: PLW0603
    if _graph is None:
        _graph = build_graph()
    return _graph
