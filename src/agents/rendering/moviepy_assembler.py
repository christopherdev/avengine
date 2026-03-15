"""
MoviePy v2.0+ Video Assembler.

Responsibility:
  Given a refined VideoTimeline, use MoviePy v2.0 to:
    1. Load and validate each source clip file.
    2. Trim each clip to its source window (source_start → source_end).
    3. Resize/pad to the target resolution.
    4. Build TextClip overlays for titles and lower-thirds.
    5. Duck B-roll audio to the configured gain level.
    6. Return an AssemblyPlan — a structured description of all clips,
       their MoviePy parameters, and composite positions.

The AssemblyPlan is NOT used to call MoviePy's write_videofile() directly.
Instead it is consumed by the FFmpegRenderer, which translates the plan into
a raw filter_complex command, bypassing the Python GIL for the final encode.

MoviePy v2.0 API changes used throughout:
  - logger=None          instead of verbose=False (deprecated)
  - clip.with_start()    instead of clip.set_start()
  - clip.with_end()      instead of clip.set_end()
  - clip.with_duration() instead of clip.set_duration()
  - clip.with_position() instead of clip.set_position()
  - clip.with_opacity()  instead of clip.set_opacity()
  - clip.with_audio()    instead of clip.set_audio()
  - clip.with_volume_scaled() instead of volumex()
  - ImageClip / TextClip constructors use keyword-only args
  - CompositeVideoClip(clips, size=...) — size is now a required kwarg
    when the first clip isn't full-frame
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.schemas.pipeline import ClipTransition, TextOverlay, TimelineClip, VideoTimeline

logger = structlog.get_logger(__name__)


# ── Assembly plan data structures ─────────────────────────────────────────────

@dataclass
class ClipSpec:
    """
    All parameters needed to process one TimelineClip for the final render.
    """
    clip_id: str
    local_path: str
    source_start: float
    source_end: float
    timeline_start: float
    timeline_end: float
    target_width: int
    target_height: int
    audio_gain_db: float
    transition_in: ClipTransition
    transition_out: ClipTransition
    overlays: list[OverlaySpec] = field(default_factory=list)
    loop_source: bool = False       # True when source is shorter than required duration
    needs_scale: bool = False
    has_audio: bool = True          # False for video-only clips (no audio stream)
    source_width: int = 0
    source_height: int = 0
    source_fps: float = 30.0


@dataclass
class OverlaySpec:
    """Computed parameters for a TextClip overlay."""
    text: str
    x: int
    y: int
    width: int
    font_size: int
    color: str
    bg_color: str | None
    start_seconds: float     # relative to clip start
    end_seconds: float       # relative to clip start (None = full clip)
    font_path: str | None


@dataclass
class AudioTrackSpec:
    """Parameters for an audio track (narration or music)."""
    track_id: str
    local_path: str
    timeline_start: float
    volume: float
    fade_in: float
    fade_out: float
    total_duration: float


@dataclass
class AssemblyPlan:
    """
    Complete description of the video assembly.
    Consumed by FFmpegRenderer to build the filter_complex command.
    """
    task_id: str
    output_width: int
    output_height: int
    fps: float
    total_duration: float
    clips: list[ClipSpec]
    audio_tracks: list[AudioTrackSpec]
    output_path: str


# ── Assembler ─────────────────────────────────────────────────────────────────

class MoviePyAssembler:
    """
    Analyses the VideoTimeline with MoviePy to produce an AssemblyPlan.

    MoviePy is used here purely for measurement and validation — the
    actual pixel-pushing encode is done by FFmpegRenderer via subprocess.
    """

    def build_plan(
        self,
        timeline: VideoTimeline,
        output_path: str,
    ) -> AssemblyPlan:
        """
        Walk every TimelineClip in the timeline, probe source files,
        and produce ClipSpec + OverlaySpec objects.
        """
        from moviepy import AudioFileClip, VideoFileClip

        clip_specs: list[ClipSpec] = []
        audio_specs: list[AudioTrackSpec] = []

        target_w = timeline.output_width
        target_h = timeline.output_height

        for tc in timeline.clips:
            spec = self._build_clip_spec(tc, target_w, target_h)
            if spec is not None:
                clip_specs.append(spec)

        for at in timeline.audio_tracks:
            audio_specs.append(
                AudioTrackSpec(
                    track_id=at.track_id,
                    local_path=at.local_path,
                    timeline_start=at.timeline_start,
                    volume=at.volume,
                    fade_in=at.fade_in_seconds,
                    fade_out=at.fade_out_seconds,
                    total_duration=timeline.total_duration,
                )
            )

        logger.info(
            "assembler.plan_built",
            clips=len(clip_specs),
            audio_tracks=len(audio_specs),
            duration=round(timeline.total_duration, 2),
        )

        return AssemblyPlan(
            task_id=timeline.task_id,
            output_width=target_w,
            output_height=target_h,
            fps=timeline.target_fps,
            total_duration=timeline.total_duration,
            clips=clip_specs,
            audio_tracks=audio_specs,
            output_path=output_path,
        )

    # ── Clip spec builder ──────────────────────────────────────────────────────

    def _build_clip_spec(
        self,
        tc: TimelineClip,
        target_w: int,
        target_h: int,
    ) -> ClipSpec | None:
        """Probe the source file and compute all rendering parameters."""
        from moviepy import VideoFileClip

        path = pathlib.Path(tc.local_path)
        if not path.exists() or tc.local_path in ("/dev/null", ""):
            logger.warning("assembler.missing_file", path=str(path))
            return None

        try:
            probe = VideoFileClip(str(path))
            src_w, src_h = probe.size
            src_fps = probe.fps or 30.0
            src_duration = probe.duration or 0.0
            has_audio = probe.audio is not None
            probe.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("assembler.probe_failed", path=str(path), error=str(exc))
            return None

        required_duration = tc.timeline_end - tc.timeline_start
        available = tc.source_end - tc.source_start
        loop_source = available < required_duration

        needs_scale = (src_w != target_w or src_h != target_h)

        # Build overlay specs
        overlays = [
            self._build_overlay_spec(ov, target_w, target_h, required_duration)
            for ov in tc.overlays
        ]

        return ClipSpec(
            clip_id=tc.clip_id,
            local_path=str(path),
            source_start=tc.source_start,
            source_end=min(tc.source_end, src_duration),
            timeline_start=tc.timeline_start,
            timeline_end=tc.timeline_end,
            target_width=target_w,
            target_height=target_h,
            audio_gain_db=tc.audio_gain_db,
            transition_in=tc.transition_in,
            transition_out=tc.transition_out,
            overlays=overlays,
            loop_source=loop_source,
            needs_scale=needs_scale,
            has_audio=has_audio,
            source_width=src_w,
            source_height=src_h,
            source_fps=src_fps,
        )

    # ── Overlay spec builder ───────────────────────────────────────────────────

    def _build_overlay_spec(
        self,
        ov: TextOverlay,
        target_w: int,
        target_h: int,
        clip_duration: float,
    ) -> OverlaySpec:
        """
        Translate a TextOverlay schema into pixel-precise OverlaySpec.

        Position mapping:
          top          → y = 5%  of height
          center       → y = 50% (vertically centered)
          bottom       → y = 90% of height
          lower_third  → y = 75% of height

        Font size is scaled proportionally to the output width so that text
        designed for 1920px stays readable on 1080px-wide reels and squares.
        Text is word-wrapped to prevent overflow on narrow outputs.
        """
        MARGIN = int(target_w * 0.05)
        overlay_w = target_w - 2 * MARGIN

        position_y_map = {
            "top":         int(target_h * 0.05),
            "center":      int(target_h * 0.45),
            "bottom":      int(target_h * 0.88),
            "lower_third": int(target_h * 0.75),
        }
        y = position_y_map.get(ov.position.value if hasattr(ov.position, 'value') else str(ov.position), int(target_h * 0.88))

        # Scale font proportionally to output width (baseline: 1920px)
        scaled_font_size = max(24, int(ov.font_size * target_w / 1920))

        # Word-wrap: approx 0.5 * font_size pixels per character
        chars_per_line = max(10, int(overlay_w / (scaled_font_size * 0.5)))
        wrapped_text = _wrap_text(ov.text, chars_per_line)

        return OverlaySpec(
            text=wrapped_text,
            x=MARGIN,
            y=y,
            width=overlay_w,
            font_size=scaled_font_size,
            color=ov.color,
            bg_color=ov.background_color,
            start_seconds=ov.start_seconds,
            end_seconds=ov.end_seconds if ov.end_seconds is not None else clip_duration,
            font_path=ov.font_path,
        )

    # ── MoviePy v2.0 preview render (optional, dev-only) ──────────────────────

    def render_preview(
        self,
        timeline: VideoTimeline,
        output_path: str,
        max_duration: float = 10.0,
    ) -> str:
        """
        Render a short preview using pure MoviePy v2.0 (no FFmpeg subprocess).
        Used for development and integration testing only.
        NOT used in the production render path.
        """
        from moviepy import (
            AudioFileClip,
            CompositeVideoClip,
            TextClip,
            VideoFileClip,
            concatenate_videoclips,
        )

        clips_mv = []
        for tc in timeline.clips[:5]:   # cap at 5 clips for preview
            p = pathlib.Path(tc.local_path)
            if not p.exists():
                continue
            try:
                clip = (
                    VideoFileClip(str(p))
                    .subclipped(tc.source_start, tc.source_end)
                    .resized((timeline.output_width, timeline.output_height))
                    .with_audio(
                        VideoFileClip(str(p))
                        .subclipped(tc.source_start, tc.source_end)
                        .audio
                        .with_volume_scaled(
                            _db_to_linear(tc.audio_gain_db)
                        ) if tc.audio_gain_db < 0 else None
                    )
                )
                clips_mv.append(clip)
            except Exception as exc:  # noqa: BLE001
                logger.warning("assembler.preview_clip_failed", error=str(exc))

        if not clips_mv:
            raise RuntimeError("No valid clips for preview render.")

        final = concatenate_videoclips(clips_mv, method="compose")
        final = final.subclipped(0, min(final.duration, max_duration))

        # Add narration if available
        if timeline.audio_tracks:
            narration_path = timeline.audio_tracks[0].local_path
            if pathlib.Path(narration_path).exists():
                narration = (
                    AudioFileClip(narration_path)
                    .subclipped(0, final.duration)
                )
                final = final.with_audio(narration)

        final.write_videofile(
            output_path,
            fps=timeline.target_fps,
            codec="libx264",
            audio_codec="aac",
            logger=None,   # MoviePy v2.0 — replaces verbose=False
        )

        for c in clips_mv:
            c.close()

        return output_path


def _db_to_linear(db: float) -> float:
    """Convert dB gain to linear scale factor."""
    return 10 ** (db / 20)


def _wrap_text(text: str, max_chars_per_line: int) -> str:
    """
    Wrap text at word boundaries.  Returns a string with newline characters
    at wrap points — the FFmpeg renderer converts these to drawtext \\n escapes.
    """
    words = text.split()
    if not words:
        return text

    lines: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        word_len = len(word)
        # +1 for the space before the word (unless it's the first word)
        needed = word_len + (1 if current else 0)
        if current and length + needed > max_chars_per_line:
            lines.append(" ".join(current))
            current = [word]
            length = word_len
        else:
            current.append(word)
            length += needed

    if current:
        lines.append(" ".join(current))

    return "\n".join(lines)
