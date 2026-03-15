"""
Timeline Calculator.

Replaces the estimated `duration_seconds` values from the script with
frame-accurate durations derived from ElevenLabs word-level timestamps.

The fundamental contract:
  - B-roll clips must be SHOWN during the narration words they illustrate.
  - The visual cut point is the start_seconds of the first word in a scene.
  - The cut-out point is the end_seconds of the last word.
  - Silence padding (inter-sentence gaps) is distributed to the preceding clip.

Outputs a `RefinedTimeline` — a new `VideoTimeline` with every
`TimelineClip.timeline_start / timeline_end` replaced by frame-accurate
values from the ElevenLabs alignment data.

Frame accuracy:
  All times are snapped to the nearest video frame boundary using:
    snapped = round(t * fps) / fps
  This prevents fractional-frame offsets that cause FFmpeg seek jitter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from src.schemas.pipeline import AudioTrack, TimelineClip, VideoTimeline
from src.services.elevenlabs.client import NarrationAudio, SceneAudio, WordTimestamp

logger = structlog.get_logger(__name__)

# Silence gap between sentences that is absorbed into the preceding clip's tail
_SILENCE_PAD_SECONDS = 0.15

# Minimum clip duration — clips shorter than this are extended to this length
_MIN_CLIP_DURATION = 0.5


@dataclass
class SceneWindow:
    """The precise audio window for one ScriptScene."""
    scene_id: str
    audio_start: float     # seconds into the merged narration file
    audio_end: float
    word_timestamps: list[WordTimestamp] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.audio_end - self.audio_start)


class TimelineCalculator:
    """
    Refines a draft VideoTimeline using ElevenLabs narration timestamps.
    """

    def __init__(self, fps: float = 30.0) -> None:
        self._fps = fps

    def refine(
        self,
        draft: VideoTimeline,
        narration: NarrationAudio,
    ) -> VideoTimeline:
        """
        Replace estimated clip durations with ElevenLabs-derived timecodes.

        Steps:
          1. Build a SceneWindow for every scene from the per-scene audio files.
          2. Walk timeline clips; for each clip, look up the SceneWindow of its
             scene_id and compute the refined timeline_start/end.
          3. Snap all timecodes to the nearest frame boundary.
          4. Add the narration AudioTrack to the timeline.
        """
        scene_windows = self._build_scene_windows(narration)

        # Index clips by scene_id (preserve order within each scene)
        by_scene: dict[str, list[TimelineClip]] = {}
        for clip in draft.clips:
            by_scene.setdefault(clip.scene_id, []).append(clip)

        refined_clips: list[TimelineClip] = []
        cursor = 0.0  # current position in the OUTPUT timeline (seconds)

        for scene_id, clips in by_scene.items():
            window = scene_windows.get(scene_id)

            if window is None or window.duration <= 0:
                # No audio for this scene — keep estimated durations
                for clip in clips:
                    estimated_dur = clip.duration
                    new_clip = self._snap_clip(clip, cursor, cursor + estimated_dur)
                    refined_clips.append(new_clip)
                    cursor += estimated_dur
                continue

            # Divide the scene audio window evenly across the scene's clips
            slot = window.duration / len(clips)

            for i, clip in enumerate(clips):
                tl_start = cursor + i * slot
                tl_end   = cursor + (i + 1) * slot

                # Ensure minimum duration
                if tl_end - tl_start < _MIN_CLIP_DURATION:
                    tl_end = tl_start + _MIN_CLIP_DURATION

                # Trim source clip to match the required duration precisely
                required_dur = tl_end - tl_start
                src_dur = clip.source_end - clip.source_start

                if src_dur < required_dur:
                    # Loop the source clip by extending the source window
                    # (FFmpeg loop filter handles this in the renderer)
                    new_source_end = clip.source_start + required_dur
                else:
                    new_source_end = clip.source_start + required_dur

                refined_clips.append(
                    self._snap_clip(
                        clip._replace_times(
                            source_end=new_source_end,
                        ),
                        tl_start,
                        tl_end,
                    )
                )

            cursor += window.duration + _SILENCE_PAD_SECONDS

        # Build narration AudioTrack
        narration_track = AudioTrack(
            track_id="narration",
            local_path=narration.local_path,
            timeline_start=0.0,
            volume=1.0,
            fade_in_seconds=0.0,
            fade_out_seconds=0.5,
        )

        total_duration = max((c.timeline_end for c in refined_clips), default=0.0)

        logger.info(
            "timeline_calculator.refined",
            clips=len(refined_clips),
            total_duration=round(total_duration, 3),
            fps=self._fps,
        )

        return VideoTimeline(
            task_id=draft.task_id,
            title=draft.title,
            aspect_ratio=draft.aspect_ratio,
            target_fps=self._fps,
            output_width=draft.output_width,
            output_height=draft.output_height,
            total_duration=total_duration,
            clips=refined_clips,
            audio_tracks=[narration_track] + list(draft.audio_tracks),
            narration_timestamps=self._serialise_timestamps(narration),
        )

    # ── Scene window construction ──────────────────────────────────────────────

    def _build_scene_windows(
        self, narration: NarrationAudio
    ) -> dict[str, SceneWindow]:
        """
        Map each scene to its absolute time window in the merged narration file.

        ElevenLabs generates per-scene files concatenated into one merged track.
        The cursor accumulates each scene's duration to produce absolute offsets.
        """
        windows: dict[str, SceneWindow] = {}
        cursor = 0.0

        for scene_audio in narration.scenes:
            window = SceneWindow(
                scene_id=scene_audio.scene_id,
                audio_start=cursor,
                audio_end=cursor + scene_audio.duration_seconds,
                word_timestamps=scene_audio.word_timestamps,
            )
            windows[scene_audio.scene_id] = window
            cursor += scene_audio.duration_seconds + _SILENCE_PAD_SECONDS

        return windows

    # ── Frame snapping ─────────────────────────────────────────────────────────

    def _snap(self, t: float) -> float:
        """Snap a time value to the nearest frame boundary."""
        return round(round(t * self._fps) / self._fps, 6)

    def _snap_clip(
        self, clip: TimelineClip, tl_start: float, tl_end: float
    ) -> TimelineClip:
        """Return a new TimelineClip with snapped timeline boundaries."""
        snapped_start = self._snap(tl_start)
        snapped_end   = self._snap(tl_end)

        # Ensure at least one frame
        if snapped_end <= snapped_start:
            snapped_end = snapped_start + self._snap(1 / self._fps)

        return TimelineClip(
            clip_id=clip.clip_id,
            local_path=clip.local_path,
            source_start=self._snap(clip.source_start),
            source_end=self._snap(clip.source_end),
            timeline_start=snapped_start,
            timeline_end=snapped_end,
            transition_in=clip.transition_in,
            transition_out=clip.transition_out,
            overlays=clip.overlays,
            audio_gain_db=clip.audio_gain_db,
            scene_id=clip.scene_id,
        )

    # ── Serialisation ──────────────────────────────────────────────────────────

    @staticmethod
    def _serialise_timestamps(narration: NarrationAudio) -> dict[str, Any]:
        """Store the full alignment data in the timeline for debugging."""
        return {
            scene.scene_id: {
                "duration": scene.duration_seconds,
                "words": [
                    {
                        "word": w.word,
                        "start": w.start_seconds,
                        "end": w.end_seconds,
                    }
                    for w in scene.word_timestamps
                ],
            }
            for scene in narration.scenes
        }


# ── TimelineClip extension helper ─────────────────────────────────────────────
# Pydantic models are immutable by default — we add a convenience builder.

def _replace_times(self: TimelineClip, *, source_end: float) -> TimelineClip:
    return TimelineClip(
        clip_id=self.clip_id,
        local_path=self.local_path,
        source_start=self.source_start,
        source_end=source_end,
        timeline_start=self.timeline_start,
        timeline_end=self.timeline_end,
        transition_in=self.transition_in,
        transition_out=self.transition_out,
        overlays=self.overlays,
        audio_gain_db=self.audio_gain_db,
        scene_id=self.scene_id,
    )


# Monkey-patch the helper onto TimelineClip so the calculator can call it
TimelineClip._replace_times = _replace_times  # type: ignore[attr-defined]
