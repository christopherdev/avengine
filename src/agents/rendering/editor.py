"""
VideoEditor — full rendering pipeline implementation.

Orchestrates the complete rendering chain:

  1. ElevenLabs TTS       → per-scene MP3 files + word timestamps
  2. TimelineCalculator   → refine VideoTimeline with frame-accurate durations
  3. MoviePyAssembler     → probe source files, build AssemblyPlan
  4. FFmpegRenderer       → execute filter_complex render → output.mp4
  5. FFmpegRenderer       → extract thumbnail JPEG

All steps run synchronously inside the Celery `render` queue worker.
"""
from __future__ import annotations

import pathlib
from typing import Any

import structlog

from src.agents.rendering.ffmpeg_renderer import FFmpegRenderer
from src.agents.rendering.moviepy_assembler import MoviePyAssembler
from src.agents.rendering.timeline_calculator import TimelineCalculator
from src.core.config import get_settings
from src.core.exceptions import RenderingError
from src.schemas.pipeline import VideoTimeline

logger = structlog.get_logger(__name__)
settings = get_settings()


class VideoEditor:
    """
    Entry point for the LangGraph rendering node.
    Wires all rendering sub-components together.
    """

    def __init__(self) -> None:
        self._assembler  = MoviePyAssembler()
        self._renderer   = FFmpegRenderer()
        self._calculator = TimelineCalculator()

    # ── Public API ─────────────────────────────────────────────────────────────

    def render_sync(
        self,
        task_id: str,
        timeline_dict: dict[str, Any],
    ) -> tuple[str, str]:
        """
        Full render pipeline.
        Returns (video_path, thumbnail_path).
        """
        log = logger.bind(task_id=task_id)
        log.info("editor.render_start")

        timeline = VideoTimeline(**timeline_dict)

        output_dir = pathlib.Path(settings.video_output_dir) / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path    = str(output_dir / "output.mp4")
        thumbnail_path = str(output_dir / "thumbnail.jpg")

        try:
            # ── Step 1: TTS narration ─────────────────────────────────────────
            narration = self._synthesise_narration(task_id, timeline, log)

            # ── Step 2: Refine timeline with word timestamps ───────────────────
            if narration is not None:
                timeline = self._calculator.refine(timeline, narration)
                log.info(
                    "editor.timeline_refined",
                    clips=len(timeline.clips),
                    duration=round(timeline.total_duration, 2),
                )

            # ── Step 3: Build assembly plan ───────────────────────────────────
            plan = self._assembler.build_plan(timeline, output_path)

            if not plan.clips:
                raise RenderingError(
                    "No renderable clips in the assembly plan.",
                    detail=f"task_id={task_id}",
                )

            # ── Step 4: FFmpeg render ─────────────────────────────────────────
            log.info("editor.ffmpeg_start", clips=len(plan.clips))
            self._renderer.render(plan)
            log.info("editor.ffmpeg_done", output=output_path)

            # ── Step 5: Thumbnail ─────────────────────────────────────────────
            thumb_at = min(2.0, timeline.total_duration * 0.1)
            self._renderer.render_thumbnail(output_path, thumbnail_path, at_second=thumb_at)

        except RenderingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderingError(
                f"Rendering failed for task {task_id}: {exc}",
                detail=str(exc),
            ) from exc

        log.info("editor.render_complete", video=output_path, thumb=thumbnail_path)
        return output_path, thumbnail_path

    # ── TTS synthesis ──────────────────────────────────────────────────────────

    def _synthesise_narration(
        self,
        task_id: str,
        timeline: VideoTimeline,
        log: Any,
    ) -> Any:
        """
        Call ElevenLabs to synthesise narration.
        Returns NarrationAudio, or None if ElevenLabs is not configured.
        """
        if not settings.elevenlabs_api_key:
            log.warning("editor.elevenlabs_not_configured")
            return None

        from src.services.elevenlabs.client import get_elevenlabs_client

        scenes = self._scenes_from_timeline(timeline)
        if not scenes:
            log.warning("editor.no_scenes_for_tts")
            return None

        try:
            client = get_elevenlabs_client()
            narration = client.synthesise_script(task_id=task_id, scenes=scenes)
            log.info(
                "editor.tts_done",
                scenes=len(narration.scenes),
                total_duration=round(narration.total_duration, 2),
            )
            return narration
        except Exception as exc:  # noqa: BLE001
            log.warning("editor.tts_failed", error=str(exc)[:300])
            return None

    def _scenes_from_timeline(self, timeline: VideoTimeline) -> list[dict[str, Any]]:
        """
        Build scene narration data for TTS from scene_narrations stored by
        the Matching node.  Falls back to narration_timestamps word text if
        scene_narrations is absent (e.g. old checkpoints).
        """
        if timeline.scene_narrations:
            return [
                {"scene_id": scene_id, "narration": narration}
                for scene_id, narration in timeline.scene_narrations.items()
                if narration.strip()
            ]

        # Legacy fallback: reconstruct from word-level timestamps
        timestamps: dict[str, Any] = timeline.narration_timestamps
        if not timestamps:
            return []

        scenes = []
        for scene_id, data in timestamps.items():
            words = data.get("words", [])
            narration = " ".join(w["word"] for w in words) if words else ""
            if narration:
                scenes.append({"scene_id": scene_id, "narration": narration})
        return scenes
