"""
Pydantic AI structured extraction node.

Responsibility: take the raw JSON-ish text produced by the CrewAI Editor
and guarantee it conforms to VideoScript.  Uses a multi-attempt strategy:

  1. Direct JSON parse + Pydantic validation (zero LLM calls, fastest path).
  2. Pydantic AI agent extraction — asks the LLM to emit a structured object
     when the raw text is malformed or contains markdown fences.
  3. Repair pass — asks the LLM to fix specific Pydantic validation errors.

The LLM in step 2/3 is configured with `result_type=VideoScript` so Pydantic AI
enforces the schema at the output layer, making hallucinated fields impossible.
"""
from __future__ import annotations

import json
import textwrap
from typing import Any

import structlog
from pydantic import ValidationError
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from src.core.config import get_settings
from src.schemas.pipeline import BRollCue, ScriptScene, VideoScript

logger = structlog.get_logger(__name__)


def _make_extraction_agent() -> PydanticAgent:
    """Instantiate lazily so API key is available at call time."""
    s = get_settings()
    model = AnthropicModel(
        model_name=s.anthropic_model,
        provider=AnthropicProvider(api_key=s.anthropic_api_key),
    )
    return PydanticAgent(
        model=model,
        result_type=VideoScript,
        system_prompt=textwrap.dedent("""
            You are a precise data extraction engine.  Your sole task is to parse
            the provided script text and return a fully-structured VideoScript object.

            Rules:
            - scene_id must be zero-padded strings: "S01", "S02", etc.
            - cue_id must follow the pattern "<scene_id>C<two-digit-number>": "S01C01"
            - duration_seconds must be a positive float
            - transition values must be one of: cut, fade, dissolve, wipe
            - tone must be one of: authoritative, conversational, inspirational, humorous, neutral
            - narration must be 5–1000 characters
            - Do not invent content — extract only what is present in the input
            - If a field is missing, use the schema default where available
        """).strip(),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def extract_video_script(raw_text: str, title: str = "") -> VideoScript:
    """
    Attempt to parse raw_text into a VideoScript, escalating through three
    strategies until one succeeds.

    Raises TimelineExtractionError if all strategies fail.
    """
    from src.core.exceptions import TimelineExtractionError

    # ── Strategy 1: direct parse ──────────────────────────────────────────────
    try:
        return _direct_parse(raw_text, title)
    except (json.JSONDecodeError, ValidationError, KeyError) as exc:
        logger.info("extractor.direct_parse_failed", reason=str(exc)[:200])

    # ── Strategy 2: Pydantic AI extraction ────────────────────────────────────
    try:
        return _pydantic_ai_extract(raw_text, title)
    except Exception as exc:  # noqa: BLE001
        logger.warning("extractor.pydantic_ai_failed", reason=str(exc)[:200])

    # ── Strategy 3: repair pass ───────────────────────────────────────────────
    try:
        return _repair_extract(raw_text, title)
    except Exception as exc:  # noqa: BLE001
        logger.error("extractor.all_strategies_failed", reason=str(exc)[:200])
        raise TimelineExtractionError(
            "Failed to extract a valid VideoScript after 3 attempts.",
            detail=str(exc),
        ) from exc


# ── Strategy implementations ──────────────────────────────────────────────────

def _direct_parse(raw: str, title: str) -> VideoScript:
    """Try plain JSON decode + Pydantic construction."""
    cleaned = _strip_markdown(raw)
    data: dict[str, Any] = json.loads(cleaned)

    # Inject title if the LLM omitted it
    if not data.get("title") and title:
        data["title"] = title

    # Normalise scene structure
    data["scenes"] = [_normalise_scene(s, i) for i, s in enumerate(data["scenes"], 1)]

    return VideoScript(**data)


def _pydantic_ai_extract(raw: str, title: str) -> VideoScript:
    """
    Use the Pydantic AI agent to perform structured extraction.
    Runs synchronously via `run_sync`.
    """
    prompt = (
        f"Extract a VideoScript from the following script text.\n"
        f"Title hint: {title or 'derive from content'}\n\n"
        f"---\n{raw}\n---"
    )

    result = _make_extraction_agent().run_sync(prompt)
    script: VideoScript = result.data

    if not script.title and title:
        script.title = title

    logger.info(
        "extractor.pydantic_ai_success",
        scenes=len(script.scenes),
        title=script.title,
    )
    return script


def _repair_extract(raw: str, title: str) -> VideoScript:
    """
    Ask the LLM to fix a broken JSON blob by describing the errors,
    then run through Pydantic AI extraction again.
    """
    from langchain_anthropic import ChatAnthropic

    s = get_settings()
    repair_llm = ChatAnthropic(
        model=s.anthropic_model,
        temperature=0.0,
        api_key=s.anthropic_api_key,
    )

    repair_prompt = textwrap.dedent(f"""
        The following text was supposed to be a video script JSON but it has
        formatting errors.  Rewrite it as a clean JSON object matching this schema:

        {{
          "title": "string",
          "tone": "conversational",
          "scenes": [
            {{
              "scene_id": "S01",
              "narration": "string (5-1000 chars)",
              "duration_seconds": 8.0,
              "b_roll_cues": [
                {{
                  "cue_id": "S01C01",
                  "description": "string",
                  "search_query": "string",
                  "duration_seconds": 4.0,
                  "transition_in": "cut",
                  "transition_out": "cut"
                }}
              ]
            }}
          ]
        }}

        Input to fix:
        {raw[:6000]}

        Return ONLY the corrected JSON. No markdown. No explanation.
    """).strip()

    response = repair_llm.invoke(repair_prompt)
    repaired_text: str = response.content

    return _direct_parse(repaired_text, title)


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove ```json ... ``` fences and leading/trailing whitespace."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner)
    return text.strip()


def _normalise_scene(raw_scene: dict[str, Any], index: int) -> dict[str, Any]:
    """Ensure scene_id and cue_id fields are correctly formatted."""
    scene_id = raw_scene.get("scene_id") or f"S{index:02d}"
    if not scene_id.startswith("S"):
        scene_id = f"S{index:02d}"
    raw_scene["scene_id"] = scene_id

    cues = raw_scene.get("b_roll_cues", [])
    for j, cue in enumerate(cues, 1):
        if not cue.get("cue_id") or not cue["cue_id"].startswith(scene_id):
            cue["cue_id"] = f"{scene_id}C{j:02d}"
        cue.setdefault("transition_in", "cut")
        cue.setdefault("transition_out", "cut")

    raw_scene["b_roll_cues"] = cues
    raw_scene.setdefault("duration_seconds", max(
        len(raw_scene.get("narration", "").split()) / 2.2, 3.0
    ))
    return raw_scene
