"""
Domain schemas for the internal pipeline.

These are never directly exposed over HTTP — the API layer maps them to the
public schemas in schemas/api.py.  All models use Pydantic v2 syntax.

Hierarchy:
  GenerateVideoRequest  (API input)
      │
      ▼
  IdeationBrief         (ideation node output)
      │
      ▼
  VideoScript           (scripting node output)
    └── ScriptScene[]
          └── BRollCue[]
      │
      ▼
  SourcedClip[]         (sourcing node output — raw downloads)
      │
      ▼
  MatchedClip[]         (matching node output — TwelveLabs scored)
      │
      ▼
  VideoTimeline         (final editor input)
    └── TimelineClip[]
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class NarrativeTone(str, Enum):
    authoritative = "authoritative"
    conversational = "conversational"
    inspirational = "inspirational"
    humorous = "humorous"
    neutral = "neutral"


class ClipTransition(str, Enum):
    cut = "cut"
    fade = "fade"
    dissolve = "dissolve"
    wipe = "wipe"


class TextPosition(str, Enum):
    top = "top"
    center = "center"
    bottom = "bottom"
    lower_third = "lower_third"


# ── Ideation ──────────────────────────────────────────────────────────────────

class IdeationBrief(BaseModel):
    """
    Output of the Ideation node.

    Distills the raw topic into a structured brief that guides the
    CrewAI scripting crew.
    """

    title: str = Field(..., max_length=200)
    hook: str = Field(..., max_length=500, description="Opening sentence to grab attention.")
    key_points: list[str] = Field(..., min_length=3, max_length=7)
    tone: NarrativeTone = NarrativeTone.conversational
    target_audience: str = Field(default="general")
    estimated_word_count: int = Field(..., ge=50, le=2000)
    search_queries: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Queries passed to the Sourcing agent for B-roll discovery.",
    )
    style: str = Field(default="explainer")
    duration_seconds: int = Field(default=60)


# ── Script ────────────────────────────────────────────────────────────────────

class BRollCue(BaseModel):
    """
    A visual instruction embedded within a ScriptScene.

    The Sourcing agent uses `search_query` to find matching footage;
    the Matching agent refines it to exact timestamps.
    """

    cue_id: str = Field(..., description="Unique within the scene, e.g. 'S01C01'.")
    description: str = Field(..., max_length=300)
    search_query: str = Field(..., max_length=200)
    duration_seconds: float = Field(..., ge=0.5, le=30.0)
    transition_in: ClipTransition = ClipTransition.cut
    transition_out: ClipTransition = ClipTransition.cut


class ScriptScene(BaseModel):
    """
    One atomic segment of the video narrative.

    Maps 1:1 to a narration sentence (or short group of sentences) that
    ElevenLabs will synthesise.  The Matching agent fills `matched_clip_id`
    after the sourcing + matching passes complete.
    """

    scene_id: str = Field(..., description="Zero-padded scene number, e.g. 'S01'.")
    narration: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Exact words spoken by the narrator for this scene.",
    )
    duration_seconds: float = Field(
        ...,
        ge=1.0,
        le=60.0,
        description="Estimated speaking duration (recalculated from ElevenLabs timestamps later).",
    )
    b_roll_cues: list[BRollCue] = Field(default_factory=list)

    # Populated by Matching node
    matched_clip_ids: list[str] = Field(default_factory=list)

    @field_validator("scene_id")
    @classmethod
    def _validate_scene_id(cls, v: str) -> str:
        if not v.startswith("S"):
            raise ValueError("scene_id must start with 'S', e.g. 'S01'")
        return v


class VideoScript(BaseModel):
    """
    Complete narration script produced by the CrewAI Script crew.
    """

    title: str
    scenes: list[ScriptScene] = Field(..., min_length=1)
    total_estimated_duration: float = Field(default=0.0, ge=0.0)
    tone: NarrativeTone
    raw_text: str = Field(default="", description="Full concatenated narration for TTS.")

    @model_validator(mode="after")
    def _sync_total_duration(self) -> VideoScript:
        self.total_estimated_duration = sum(s.duration_seconds for s in self.scenes)
        self.raw_text = " ".join(s.narration for s in self.scenes)
        return self


# ── Sourcing ──────────────────────────────────────────────────────────────────

class SourcedClip(BaseModel):
    """
    A raw video asset downloaded by the Sourcing agent.
    """

    clip_id: str
    source_url: str
    local_path: str
    platform: str = Field(description="youtube | tiktok | instagram | rss | unknown")
    title: str = ""
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Matching ──────────────────────────────────────────────────────────────────

class MatchedClip(BaseModel):
    """
    A SourcedClip enriched with TwelveLabs semantic match data.
    """

    clip_id: str
    local_path: str
    source_url: str
    scene_id: str = Field(description="Which ScriptScene this clip serves.")
    cue_id: str = Field(description="Which BRollCue within that scene.")
    similarity_score: float = Field(..., ge=0.0, le=1.0)

    # Exact timestamps from Qdrant / TwelveLabs query
    start_seconds: float = Field(..., ge=0.0)
    end_seconds: float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def _check_timestamps(self) -> MatchedClip:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


# ── Timeline ──────────────────────────────────────────────────────────────────

class TextOverlay(BaseModel):
    """An on-screen text element rendered by MoviePy."""

    text: str
    position: TextPosition = TextPosition.lower_third
    font_size: int = Field(default=48, ge=12, le=120)
    color: str = Field(default="#FFFFFF", pattern=r"^#[0-9A-Fa-f]{6}$")
    background_color: str | None = None
    start_seconds: float = Field(default=0.0, ge=0.0)
    end_seconds: float | None = None  # None = duration of parent clip
    font_path: str | None = None


class TimelineClip(BaseModel):
    """
    One entry in the final edit timeline.

    Represents a single clip segment placed at a specific point in the
    output video, with optional text overlays and audio ducking metadata.
    """

    clip_id: str
    local_path: str
    source_start: float = Field(..., ge=0.0, description="In-point within the source file.")
    source_end: float = Field(..., ge=0.0, description="Out-point within the source file.")
    timeline_start: float = Field(..., ge=0.0, description="Placement in the output timeline.")
    timeline_end: float = Field(..., ge=0.0)
    transition_in: ClipTransition = ClipTransition.cut
    transition_out: ClipTransition = ClipTransition.cut
    overlays: list[TextOverlay] = Field(default_factory=list)
    audio_gain_db: float = Field(default=-18.0, description="B-roll audio ducked to this level.")
    scene_id: str = ""

    @model_validator(mode="after")
    def _check_timeline(self) -> TimelineClip:
        if self.timeline_end <= self.timeline_start:
            raise ValueError("timeline_end must be greater than timeline_start")
        if self.source_end <= self.source_start:
            raise ValueError("source_end must be greater than source_start")
        return self

    @property
    def duration(self) -> float:
        return self.timeline_end - self.timeline_start


class AudioTrack(BaseModel):
    """Narration or music track placed on the timeline."""

    track_id: str
    local_path: str
    timeline_start: float = 0.0
    volume: float = Field(default=1.0, ge=0.0, le=2.0)
    fade_in_seconds: float = 0.0
    fade_out_seconds: float = 0.5


class VideoTimeline(BaseModel):
    """
    The fully-resolved edit timeline consumed by the Rendering agent.

    This is the single source of truth handed from the Matching node to
    the MoviePy/FFmpeg rendering pipeline.
    """

    task_id: str
    title: str
    aspect_ratio: str = "16:9"
    target_fps: float = 30.0
    output_width: int = 1920
    output_height: int = 1080
    total_duration: float = Field(..., ge=1.0)

    clips: list[TimelineClip] = Field(..., min_length=1)
    audio_tracks: list[AudioTrack] = Field(default_factory=list)

    # Scene narration text — populated by Matching node, consumed by TTS
    scene_narrations: dict[str, str] = Field(default_factory=dict)

    # ElevenLabs word-level timestamp JSON (populated in Step 5)
    narration_timestamps: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _compute_duration(self) -> VideoTimeline:
        if self.clips:
            self.total_duration = max(c.timeline_end for c in self.clips)
        return self


# ── LangGraph Pipeline State ──────────────────────────────────────────────────

class PipelineError(BaseModel):
    node: str
    message: str
    recoverable: bool = False


class PipelineState(BaseModel):
    """
    The shared mutable state object threaded through every LangGraph node.

    LangGraph requires a TypedDict for its state, but we maintain a Pydantic
    model alongside it for validation.  The graph node functions convert
    between the two (see graph.py).

    Fields are optional so the state can be partially populated as nodes run.
    """

    # Identity
    task_id: str
    request: dict[str, Any] = Field(default_factory=dict)

    # Node outputs (None until the node runs)
    brief: IdeationBrief | None = None
    script: VideoScript | None = None
    sourced_clips: list[SourcedClip] = Field(default_factory=list)
    matched_clips: list[MatchedClip] = Field(default_factory=list)
    timeline: VideoTimeline | None = None

    # Rendered output
    output_video_path: str | None = None
    output_video_url: str | None = None
    thumbnail_url: str | None = None

    # Control flow
    current_node: str = "ideation"
    errors: list[PipelineError] = Field(default_factory=list)
    retry_count: int = 0
    is_cancelled: bool = False
