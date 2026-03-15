"""
Semantic Matcher — full implementation.

End-to-end flow per pipeline run:

  Phase 1 — Indexing
  ──────────────────
  For each SourcedClip:
    1. Upload local video file to TwelveLabs → get video_id.
    2. Call TwelveLabs Marengo 3.0 embeddings API → get segment vectors.
    3. Upsert segment vectors into Qdrant with payload:
         {clip_id, video_id, task_id, local_path, start_s, end_s, scene_id}.
    4. (Optional) Call Pegasus 1.2 to enrich each clip with a text description
       for re-ranking purposes.

  Phase 2 — Matching
  ──────────────────
  For each (ScriptScene, BRollCue) pair:
    1. Embed the scene narration sentence via TwelveLabs Marengo text embeddings.
    2. Query Qdrant with the text vector, filtered to task_id + scene_id.
    3. Re-rank results using Pegasus description cosine similarity (optional).
    4. Take the top hit → build a MatchedClip with exact start/end timestamps.

  Phase 3 — Timeline assembly
  ────────────────────────────
  Build a VideoTimeline by placing MatchedClips on a sequential timeline
  aligned to the narration word timestamps from ElevenLabs (Step 6).
  At this stage we use estimated scene durations from the script.
"""
from __future__ import annotations

import concurrent.futures
import time
from typing import Any

import structlog

from src.core.config import get_settings
from src.core.exceptions import EmbeddingError, MatchingError
from src.schemas.pipeline import (
    BRollCue,
    MatchedClip,
    ScriptScene,
    SourcedClip,
    TextOverlay,
    TimelineClip,
    VideoScript,
    VideoTimeline,
)
from src.services.twelvelabs.client import (
    TwelveLabsClient,
    VideoSegmentEmbedding,
    get_twelvelabs_client,
)
from src.services.vector_db.qdrant_service import QdrantService, get_qdrant_service

logger = structlog.get_logger(__name__)
settings = get_settings()

# Minimum cosine similarity to accept a match
MIN_MATCH_SCORE = 0.45
# How many Qdrant results to retrieve per query before re-ranking
TOP_K_RETRIEVAL = 8
# Workers for parallel clip indexing
MAX_INDEX_WORKERS = 3


class SemanticMatcher:
    """
    Orchestrates TwelveLabs embedding generation and Qdrant vector search
    to match each B-roll cue in the script to the best available clip segment.
    """

    def __init__(self) -> None:
        self._tl: TwelveLabsClient = get_twelvelabs_client()
        self._qdrant: QdrantService = get_qdrant_service()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_sync(
        self,
        task_id: str,
        script_dict: dict[str, Any],
        sourced_clips: list[dict[str, Any]],
        request: dict[str, Any],
    ) -> tuple[list[MatchedClip], VideoTimeline]:
        """
        Synchronous entry point from the LangGraph matching node.
        Returns (matched_clips, timeline).
        """
        from src.pipeline.publisher import publish_event
        from src.schemas.api import AgentStage

        log = logger.bind(task_id=task_id)
        log.info("matcher.start", clips=len(sourced_clips))

        # Deserialise
        clips = [SourcedClip(**c) for c in sourced_clips]
        script = VideoScript(**script_dict)

        if not clips:
            log.warning("matcher.no_clips")
            timeline = _build_empty_timeline(task_id, script, request)
            return [], timeline

        # ── Phase 1: Index all clips into TwelveLabs + Qdrant ─────────────────
        publish_event(
            task_id, AgentStage.matching,
            f"Indexing {len(clips)} clips into TwelveLabs...", progress=56.0,
        )
        indexed = self._index_clips(clips, task_id, script)
        log.info("matcher.indexed", indexed=len(indexed))

        # ── Phase 2: Embed script sentences + query Qdrant ────────────────────
        publish_event(
            task_id, AgentStage.matching,
            "Matching narration to video segments...", progress=63.0,
        )
        matched_clips = self._match_scenes(script, task_id, log)
        log.info("matcher.matched", matches=len(matched_clips))

        # ── Fallback: direct clip assignment when TwelveLabs unavailable ───────
        if not matched_clips:
            no_segments = all(len(segs) == 0 for segs in indexed.values())
            if no_segments and clips:
                log.warning(
                    "matcher.fallback_direct_assign",
                    reason="TwelveLabs unavailable or returned no segments",
                )
                matched_clips = _direct_assign_clips(clips, script)
                log.info("matcher.fallback_assigned", matches=len(matched_clips))

        # ── Phase 3: Build timeline ────────────────────────────────────────────
        publish_event(
            task_id, AgentStage.matching,
            "Assembling video timeline...", progress=68.0,
        )
        timeline = _build_timeline(task_id, script, matched_clips, request)

        return matched_clips, timeline

    # ── Phase 1: Indexing ──────────────────────────────────────────────────────

    def _index_clips(
        self,
        clips: list[SourcedClip],
        task_id: str,
        script: VideoScript,
    ) -> dict[str, list[VideoSegmentEmbedding]]:
        """
        Upload and embed all sourced clips in parallel.
        Returns mapping: clip_id → list[VideoSegmentEmbedding].
        """
        # Build a scene_id hint for each clip by matching clip_id prefix
        def _scene_hint(clip: SourcedClip) -> str:
            for scene in script.scenes:
                if clip.clip_id.startswith(scene.scene_id):
                    return scene.scene_id
            return ""

        results: dict[str, list[VideoSegmentEmbedding]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_INDEX_WORKERS) as executor:
            futures = {
                executor.submit(
                    self._index_one_clip, clip, task_id, _scene_hint(clip)
                ): clip
                for clip in clips
            }
            for future in concurrent.futures.as_completed(futures):
                clip = futures[future]
                try:
                    segments = future.result()
                    results[clip.clip_id] = segments
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "matcher.index_failed",
                        clip_id=clip.clip_id,
                        error=str(exc)[:200],
                    )

        return results

    def _index_one_clip(
        self,
        clip: SourcedClip,
        task_id: str,
        scene_id: str,
    ) -> list[VideoSegmentEmbedding]:
        """
        Upload one clip to TwelveLabs, extract segment embeddings, upsert to Qdrant.
        """
        log = logger.bind(clip_id=clip.clip_id, task_id=task_id)

        # Skip stub paths from earlier pipeline stages
        if clip.local_path in ("/dev/null", "") or not clip.local_path:
            log.debug("matcher.skip_stub_clip")
            return []

        try:
            video_id = self._tl.index_video(clip.local_path, clip.clip_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("matcher.index_upload_failed", error=str(exc)[:200])
            return []

        try:
            segments = self._tl.get_video_embeddings(
                video_id=video_id,
                clip_id=clip.clip_id,
                local_path=clip.local_path,
            )
        except EmbeddingError as exc:
            log.warning("matcher.embed_failed", error=str(exc)[:200])
            return []

        if segments:
            # Upsert into Qdrant
            self._qdrant.upsert_segments(segments, task_id=task_id, scene_id=scene_id)
            log.info("matcher.clip_indexed", segments=len(segments), video_id=video_id)

        return segments

    # ── Phase 2: Scene-to-clip matching ───────────────────────────────────────

    def _match_scenes(
        self,
        script: VideoScript,
        task_id: str,
        log: Any,
    ) -> list[MatchedClip]:
        """
        For each (scene, cue) pair:
          1. Build a rich query sentence combining narration + cue description.
          2. Embed the query text with TwelveLabs Marengo.
          3. Query Qdrant for the top-K most similar video segments.
          4. Select the best hit above MIN_MATCH_SCORE.
        """
        # Flatten all (scene, cue) pairs and their query texts
        query_items: list[tuple[ScriptScene, BRollCue, str]] = []
        for scene in script.scenes:
            for cue in scene.b_roll_cues:
                # Combine narration context with cue description for richer embedding
                query_text = (
                    f"{scene.narration.strip()} — {cue.description.strip()}"
                )
                query_items.append((scene, cue, query_text))

        if not query_items:
            # No B-roll cues — match whole scenes to clips
            for scene in script.scenes:
                query_items.append((scene, None, scene.narration.strip()))  # type: ignore[arg-type]

        if not query_items:
            return []

        # Batch-embed all query texts
        texts = [q[2] for q in query_items]
        try:
            text_embeddings = self._tl.embed_texts_batch(texts)
        except EmbeddingError as exc:
            log.error("matcher.batch_embed_failed", error=str(exc))
            return []

        # Build batch Qdrant queries
        qdrant_queries = [
            {
                "vector": te.embedding,
                "task_id": task_id,
                "scene_id": scene.scene_id,
            }
            for (scene, cue, _), te in zip(query_items, text_embeddings, strict=True)
        ]

        all_hits = self._qdrant.batch_query(
            queries=qdrant_queries,
            top_k=TOP_K_RETRIEVAL,
            min_score=MIN_MATCH_SCORE,
        )

        # Build MatchedClip objects from the best hit per query
        matched: list[MatchedClip] = []
        used_segments: set[str] = set()  # avoid re-using the exact same segment twice

        for (scene, cue, query_text), hits in zip(query_items, all_hits, strict=True):
            best = self._pick_best_hit(hits, used_segments)
            if best is None:
                logger.debug(
                    "matcher.no_match",
                    scene_id=scene.scene_id,
                    query=query_text[:60],
                )
                continue

            key = f"{best['clip_id']}:{best['start_seconds']}"
            used_segments.add(key)

            cue_id = cue.cue_id if cue else f"{scene.scene_id}C01"

            matched.append(
                MatchedClip(
                    clip_id=best["clip_id"],
                    local_path=best["local_path"],
                    source_url="",
                    scene_id=scene.scene_id,
                    cue_id=cue_id,
                    similarity_score=best["score"],
                    start_seconds=best["start_seconds"],
                    end_seconds=best["end_seconds"],
                )
            )
            logger.debug(
                "matcher.match",
                scene_id=scene.scene_id,
                cue_id=cue_id,
                score=round(best["score"], 3),
                start=best["start_seconds"],
                end=best["end_seconds"],
            )

        return matched

    @staticmethod
    def _pick_best_hit(
        hits: list[dict[str, Any]],
        used_segments: set[str],
    ) -> dict[str, Any] | None:
        """
        From the top-K Qdrant hits, choose the best available segment.

        Preference order:
          1. Highest cosine similarity score
          2. Not already used for another cue (avoids duplicating the exact same
             2-second clip across multiple scenes)
          3. Minimum duration of 1.5 seconds
        """
        for hit in sorted(hits, key=lambda h: h["score"], reverse=True):
            key = f"{hit['clip_id']}:{hit['start_seconds']}"
            duration = hit.get("duration", hit["end_seconds"] - hit["start_seconds"])
            if key not in used_segments and duration >= 1.5:
                return hit
        return None


# ── Direct-assignment fallback (no TwelveLabs) ───────────────────────────────

def _direct_assign_clips(
    clips: list[SourcedClip],
    script: VideoScript,
) -> list[MatchedClip]:
    """
    Fallback matcher used when TwelveLabs is unavailable.

    Assigns sourced clips to B-roll cues in order, preferring clips whose
    clip_id starts with the scene_id prefix.  Falls back to round-robin
    across all clips when no scene-specific clips exist.
    """
    # Group clips by scene prefix
    by_scene: dict[str, list[SourcedClip]] = {}
    for clip in clips:
        for scene in script.scenes:
            if clip.clip_id.startswith(scene.scene_id):
                by_scene.setdefault(scene.scene_id, []).append(clip)
                break

    matched: list[MatchedClip] = []
    clip_cycle = list(clips)  # fallback round-robin pool
    cycle_idx = 0

    for scene in script.scenes:
        scene_clips = by_scene.get(scene.scene_id, [])
        cues = scene.b_roll_cues or []
        if not cues:
            cues_iter = [None]  # type: ignore[list-item]
        else:
            cues_iter = cues  # type: ignore[assignment]

        for i, cue in enumerate(cues_iter):
            # Pick from scene-specific pool first, then fallback to round-robin
            if i < len(scene_clips):
                clip = scene_clips[i]
            else:
                clip = clip_cycle[cycle_idx % len(clip_cycle)]
                cycle_idx += 1

            if not clip.local_path or clip.local_path in ("/dev/null", ""):
                continue

            duration = clip.duration_seconds or 10.0
            cue_id = cue.cue_id if cue else f"{scene.scene_id}C01"
            matched.append(
                MatchedClip(
                    clip_id=clip.clip_id,
                    local_path=clip.local_path,
                    source_url=clip.source_url or "",
                    scene_id=scene.scene_id,
                    cue_id=cue_id,
                    similarity_score=0.0,
                    start_seconds=0.0,
                    end_seconds=duration,
                )
            )

    return matched


# ── Timeline builder ──────────────────────────────────────────────────────────

def _build_timeline(
    task_id: str,
    script: VideoScript,
    matched_clips: list[MatchedClip],
    request: dict[str, Any],
) -> VideoTimeline:
    """
    Assemble a VideoTimeline from matched clips.

    Placement algorithm:
      - Walk scenes in order.
      - For each scene, find all MatchedClips for that scene_id.
      - Place clips sequentially, each occupying `scene.duration_seconds`
        of timeline real estate (split evenly if multiple cues).
      - Narration duration is the authoritative timecode source — clips
        are stretched or trimmed to fill the scene window.
      - Text overlays (titles, lower thirds) are added for the first scene.

    The ElevenLabs word-level timestamps (Step 6) will refine these
    estimated durations in the rendering node.
    """
    aspect_ratio = request.get("aspect_ratio", "16:9")
    if aspect_ratio == "9:16":
        width, height = 1080, 1920
    elif aspect_ratio == "1:1":
        width = height = 1080
    else:
        width, height = 1920, 1080

    # Index matched clips by scene_id
    by_scene: dict[str, list[MatchedClip]] = {}
    for mc in matched_clips:
        by_scene.setdefault(mc.scene_id, []).append(mc)

    timeline_clips: list[TimelineClip] = []
    cursor = 0.0  # current position in the output timeline (seconds)

    for scene in script.scenes:
        scene_clips = by_scene.get(scene.scene_id, [])
        scene_duration = scene.duration_seconds

        if not scene_clips:
            # No matched clip for this scene — advance cursor by scene duration
            cursor += scene_duration
            continue

        # Divide scene duration evenly across cues
        slot = scene_duration / len(scene_clips)

        for idx, mc in enumerate(scene_clips):
            clip_src_duration = mc.end_seconds - mc.start_seconds
            # Trim source to fit the slot (don't stretch — letterboxing is renderer's job)
            actual_end = mc.start_seconds + min(clip_src_duration, slot)

            overlays: list[TextOverlay] = []
            # Add a title overlay on the very first clip
            if cursor == 0.0 and idx == 0:
                overlays.append(
                    TextOverlay(
                        text=script.title,
                        position="center",
                        font_size=72,
                        color="#FFFFFF",
                        background_color="#000000",
                        start_seconds=0.0,
                        end_seconds=4.0,
                    )
                )

            timeline_clips.append(
                TimelineClip(
                    clip_id=mc.clip_id,
                    local_path=mc.local_path,
                    source_start=mc.start_seconds,
                    source_end=actual_end,
                    timeline_start=cursor,
                    timeline_end=cursor + slot,
                    scene_id=scene.scene_id,
                    overlays=overlays,
                    audio_gain_db=-18.0,
                )
            )
            cursor += slot

    # If no clips were matched at all, return an empty-but-valid timeline
    if not timeline_clips:
        return _build_empty_timeline(task_id, script, request)

    scene_narrations = {s.scene_id: s.narration for s in script.scenes}

    return VideoTimeline(
        task_id=task_id,
        title=script.title,
        aspect_ratio=aspect_ratio,
        target_fps=30.0,
        output_width=width,
        output_height=height,
        total_duration=cursor,
        clips=timeline_clips,
        scene_narrations=scene_narrations,
    )


def _build_empty_timeline(
    task_id: str,
    script: VideoScript,
    request: dict[str, Any],
) -> VideoTimeline:
    """Fallback timeline with a single black-frame placeholder clip."""
    from src.schemas.pipeline import TimelineClip

    placeholder = TimelineClip(
        clip_id="placeholder",
        local_path="/dev/null",
        source_start=0.0,
        source_end=1.0,
        timeline_start=0.0,
        timeline_end=script.total_estimated_duration,
        scene_id="S01",
    )
    return VideoTimeline(
        task_id=task_id,
        title=script.title,
        total_duration=script.total_estimated_duration,
        clips=[placeholder],
    )
