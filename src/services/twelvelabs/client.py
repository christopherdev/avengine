"""
TwelveLabs Service Client.

Wraps the TwelveLabs SDK to expose two models:

  Marengo 3.0  — multimodal embedding engine.
                 Generates 1024-dim float vectors from both video segments
                 and raw text strings.  Used for semantic search (clip ↔ script).

  Pegasus 1.2  — video understanding / generative model.
                 Used to produce rich text descriptions of indexed video
                 segments for metadata enrichment and re-ranking.

The TwelveLabs "index" is the container for all video assets.  A single
shared index is used for the entire AVEngine deployment (configured via
TWELVELABS_INDEX_ID env var).  Videos are uploaded per pipeline run and
cleaned up after rendering completes.

SDK reference: https://docs.twelvelabs.io/docs/python-sdk
"""
from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass
from typing import Any

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from twelvelabs import TwelveLabs

from src.core.config import get_settings
from src.core.exceptions import EmbeddingError, TwelveLabsError

logger = structlog.get_logger(__name__)

# Marengo 3.0 embedding dimension
EMBEDDING_DIM = 1024

# Engines — SDK v1.2.0 model name format (no dashes, no "retrieval" prefix)
ENGINE_MARENGO = "marengo3.0"   # visual + text embeddings
ENGINE_PEGASUS  = "pegasus1.2"  # video understanding / generation

# Polling interval for indexing tasks
_INDEX_POLL_INTERVAL = 5   # seconds
_INDEX_TIMEOUT       = 600 # 10 minutes


@dataclass
class VideoSegmentEmbedding:
    """A single time-windowed video segment with its Marengo embedding."""
    video_id: str
    clip_id: str          # our SourcedClip ID
    local_path: str
    start_seconds: float
    end_seconds: float
    embedding: list[float]   # 1024-dim
    confidence: float = 1.0


@dataclass
class TextEmbedding:
    """An embedding produced from a text string (script narration sentence)."""
    text: str
    embedding: list[float]   # 1024-dim


@dataclass
class VideoDescription:
    """Pegasus-generated description of a video segment."""
    video_id: str
    start_seconds: float
    end_seconds: float
    description: str
    tags: list[str]


class TwelveLabsClient:
    """
    Synchronous TwelveLabs service wrapper.

    Designed to run inside a Celery worker (sync context).
    Async convenience methods are provided via `asyncio.run()` shims
    for use from the LangGraph node.
    """

    def __init__(self) -> None:
        s = get_settings()
        if not s.twelvelabs_api_key:
            raise TwelveLabsError("TWELVELABS_API_KEY is not configured.")
        self._client = TwelveLabs(api_key=s.twelvelabs_api_key)
        self._index_id = s.twelvelabs_index_id

    # ── Index management ───────────────────────────────────────────────────────

    def ensure_index(self) -> str:
        """
        Return the configured index ID, creating the index if it doesn't
        yet exist.  Safe to call multiple times (idempotent).
        """
        if self._index_id:
            return self._index_id

        logger.info("twelvelabs.creating_index")
        index = self._client.indexes.create(
            index_name="avengine-broll",
            models=[
                {"model_name": ENGINE_MARENGO, "model_options": ["visual", "audio"]},
                {"model_name": ENGINE_PEGASUS,  "model_options": ["visual", "audio"]},
            ],
        )
        self._index_id = index.id
        logger.info("twelvelabs.index_created", index_id=self._index_id)
        return self._index_id

    # ── Video indexing ─────────────────────────────────────────────────────────

    def index_video(self, local_path: str, clip_id: str) -> str:
        """
        Upload and index a local video file.

        Blocks until the TwelveLabs indexing task completes (or times out).
        Returns the TwelveLabs video_id.
        """
        index_id = self.ensure_index()
        path = pathlib.Path(local_path)

        if not path.exists():
            raise TwelveLabsError(f"Video file not found: {local_path}")

        logger.info("twelvelabs.indexing", clip_id=clip_id, path=str(path))

        with path.open("rb") as fh:
            task = self._client.tasks.create(
                index_id=index_id,
                video_file=fh,
            )

        result = self._client.tasks.wait_for_done(
            task.id,
            sleep_interval=_INDEX_POLL_INTERVAL,
            callback=lambda t: logger.debug(
                "twelvelabs.task_pending", task_id=t.id, status=t.status
            ),
        )

        if result.status not in ("ready", "indexed"):
            raise TwelveLabsError(
                f"TwelveLabs indexing task {task.id} failed with status '{result.status}'."
            )
        if result.video_id is None:
            raise TwelveLabsError(f"Task {task.id} ready but video_id is None.")

        video_id = result.video_id
        logger.info("twelvelabs.indexed", clip_id=clip_id, video_id=video_id)
        return video_id

    # ── Video segment embeddings (Marengo) ─────────────────────────────────────

    def get_video_embeddings(
        self, video_id: str, clip_id: str, local_path: str
    ) -> list[VideoSegmentEmbedding]:
        """
        Retrieve Marengo 3.0 embeddings for all detected segments of an
        indexed video.

        Each segment is a time window the TwelveLabs engine chose as
        semantically coherent (typically 2–8 seconds).

        Returns a list of VideoSegmentEmbedding — one per segment.
        """
        path = pathlib.Path(local_path)
        if not path.exists():
            raise TwelveLabsError(f"Video file not found: {local_path}")

        try:
            with path.open("rb") as fh:
                embed_task = self._client.embed.tasks.create(
                    model_name=ENGINE_MARENGO,
                    video_file=fh,
                    video_embedding_scope=["clip"],
                )
            embed_result = self._client.embed.tasks.wait_for_done(
                embed_task.id,
                sleep_interval=3,
                callback=lambda t: logger.debug(
                    "twelvelabs.embed_task_pending",
                    task_id=t.id,
                    status=t.status,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"Failed to generate video embeddings for clip_id={clip_id}",
                detail=str(exc),
            ) from exc

        segments: list[VideoSegmentEmbedding] = []
        video_embedding = getattr(embed_result, "video_embedding", None)
        raw_segments = getattr(video_embedding, "segments", None) or []

        for seg in raw_segments:
            vector = seg.float_
            if not vector or len(vector) != EMBEDDING_DIM:
                logger.warning(
                    "twelvelabs.bad_segment_vector",
                    clip_id=clip_id,
                    dim=len(vector) if vector else 0,
                )
                continue

            segments.append(
                VideoSegmentEmbedding(
                    video_id=video_id,
                    clip_id=clip_id,
                    local_path=local_path,
                    start_seconds=seg.start_offset_sec or 0.0,
                    end_seconds=seg.end_offset_sec or 0.0,
                    embedding=vector,
                    confidence=1.0,
                )
            )

        logger.info(
            "twelvelabs.segments_extracted",
            clip_id=clip_id,
            segments=len(segments),
        )
        return segments

    # ── Text embeddings (Marengo) ─────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def embed_text(self, text: str) -> TextEmbedding:
        """
        Generate a Marengo 3.0 text embedding for a script sentence.

        The resulting 1024-dim vector is in the same space as the video
        segment vectors, enabling direct cosine similarity comparison.
        """
        try:
            result = self._client.embed.create(
                model_name=ENGINE_MARENGO,
                text=text,
            )
            text_emb = result.text_embedding
            segs = getattr(text_emb, "segments", None) or []
            if not segs:
                raise EmbeddingError(
                    "TwelveLabs returned no text embedding segments.",
                    detail=f"text={text[:100]}",
                )
            vector: list[float] = segs[0].float_ or []
            if not vector or len(vector) != EMBEDDING_DIM:
                raise EmbeddingError(
                    f"Unexpected embedding dimension: {len(vector)}",
                    detail=f"expected={EMBEDDING_DIM}",
                )
            return TextEmbedding(text=text, embedding=vector)
        except EmbeddingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"TwelveLabs text embedding failed for: {text[:100]}",
                detail=str(exc),
            ) from exc

    def embed_texts_batch(self, texts: list[str]) -> list[TextEmbedding]:
        """Embed a list of texts, respecting rate limits with brief sleeps."""
        results: list[TextEmbedding] = []
        for i, text in enumerate(texts):
            emb = self.embed_text(text)
            results.append(emb)
            if i < len(texts) - 1:
                time.sleep(0.2)  # 5 RPS conservative rate limit
        return results

    # ── Pegasus: video understanding ──────────────────────────────────────────

    def describe_video(self, video_id: str) -> VideoDescription:
        """
        Use Pegasus 1.2 to generate a natural-language description and tags
        for an indexed video.  Used for metadata enrichment and re-ranking.
        """
        try:
            result = self._client.summarize(
                video_id=video_id,
                type="summary",
                prompt=(
                    "Describe this video clip concisely in 2-3 sentences, "
                    "focusing on visual content and action. "
                    "Then list 5 descriptive tags."
                ),
            )
            raw_text: str = result.summary or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("twelvelabs.describe_failed", video_id=video_id, error=str(exc))
            return VideoDescription(
                video_id=video_id,
                start_seconds=0.0,
                end_seconds=0.0,
                description="",
                tags=[],
            )

        # Split description from tags (best-effort)
        lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]
        tags: list[str] = []
        description = raw_text

        for i, line in enumerate(lines):
            if line.lower().startswith(("tags:", "tag:")):
                tag_text = line.split(":", 1)[-1]
                tags = [t.strip().lstrip("#") for t in tag_text.split(",")]
                description = " ".join(lines[:i])
                break

        return VideoDescription(
            video_id=video_id,
            start_seconds=0.0,
            end_seconds=0.0,
            description=description,
            tags=tags[:10],
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def delete_video(self, video_id: str) -> None:
        """Remove a video from the TwelveLabs index after rendering."""
        try:
            self._client.indexes.videos.delete(
                index_id=self.ensure_index(),
                video_id=video_id,
            )
            logger.info("twelvelabs.video_deleted", video_id=video_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("twelvelabs.delete_failed", video_id=video_id, error=str(exc))


# ── Module-level singleton ────────────────────────────────────────────────────

_client: TwelveLabsClient | None = None


def get_twelvelabs_client() -> TwelveLabsClient:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = TwelveLabsClient()
    return _client
