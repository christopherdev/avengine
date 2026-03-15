"""
Qdrant Vector Database Service.

Manages a single collection — `video_embeddings` — that stores Marengo 3.0
segment vectors alongside rich payload metadata.

Collection schema
─────────────────
Vector:
  name        : "marengo"
  size        : 1024
  distance    : Cosine

Payload (indexed fields for filtered queries):
  clip_id       (keyword)  — our SourcedClip ID
  video_id      (keyword)  — TwelveLabs video_id
  task_id       (keyword)  — pipeline run ID (for bulk cleanup)
  local_path    (keyword)
  source_url    (keyword)
  platform      (keyword)
  start_seconds (float)
  end_seconds   (float)
  duration      (float)    — end - start
  scene_id      (keyword)  — which ScriptScene this clip is indexed for
  confidence    (float)

Points are cleaned up after the pipeline completes to keep the collection
from growing unboundedly in a shared deployment.

Both sync (for Celery workers) and async (for the FastAPI process) clients
are provided.
"""
from __future__ import annotations

import uuid
from typing import Any

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.core.config import get_settings
from src.core.exceptions import MatchingError
from src.services.twelvelabs.client import EMBEDDING_DIM, VideoSegmentEmbedding

logger = structlog.get_logger(__name__)
settings = get_settings()

COLLECTION = settings.qdrant_collection_name
VECTOR_NAME = "marengo"


class QdrantService:
    """
    Synchronous Qdrant service.  Used from Celery workers and LangGraph nodes.
    """

    def __init__(self) -> None:
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=30,
        )

    # ── Collection lifecycle ──────────────────────────────────────────────────

    def ensure_collection(self) -> None:
        """
        Create the `video_embeddings` collection if it does not exist.
        Idempotent — safe to call on every worker startup.
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION in existing:
            return

        self._client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                VECTOR_NAME: qmodels.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                    on_disk=True,          # offload to disk for large collections
                )
            },
            optimizers_config=qmodels.OptimizersConfigDiff(
                indexing_threshold=10_000,
                memmap_threshold=50_000,
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10_000,
            ),
        )

        # Create payload indexes for filtered queries
        for field, schema_type in [
            ("task_id",   qmodels.PayloadSchemaType.KEYWORD),
            ("clip_id",   qmodels.PayloadSchemaType.KEYWORD),
            ("video_id",  qmodels.PayloadSchemaType.KEYWORD),
            ("scene_id",  qmodels.PayloadSchemaType.KEYWORD),
            ("platform",  qmodels.PayloadSchemaType.KEYWORD),
        ]:
            self._client.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=schema_type,
            )

        logger.info("qdrant.collection_created", collection=COLLECTION)

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_segments(
        self,
        segments: list[VideoSegmentEmbedding],
        task_id: str,
        scene_id: str = "",
    ) -> list[str]:
        """
        Upsert a batch of video segment embeddings into Qdrant.

        Returns the list of point IDs inserted (UUIDs).
        """
        self.ensure_collection()

        points: list[qmodels.PointStruct] = []
        point_ids: list[str] = []

        for seg in segments:
            point_id = str(uuid.uuid4())
            duration = max(0.0, seg.end_seconds - seg.start_seconds)

            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector={VECTOR_NAME: seg.embedding},
                    payload={
                        "clip_id":       seg.clip_id,
                        "video_id":      seg.video_id,
                        "task_id":       task_id,
                        "local_path":    seg.local_path,
                        "start_seconds": seg.start_seconds,
                        "end_seconds":   seg.end_seconds,
                        "duration":      duration,
                        "scene_id":      scene_id,
                        "confidence":    seg.confidence,
                        "platform":      "",
                    },
                )
            )
            point_ids.append(point_id)

        # Qdrant recommends batches of ≤ 100 points
        BATCH = 100
        for i in range(0, len(points), BATCH):
            self._client.upsert(
                collection_name=COLLECTION,
                points=points[i : i + BATCH],
                wait=True,
            )

        logger.info(
            "qdrant.upserted",
            count=len(points),
            task_id=task_id,
            scene_id=scene_id,
        )
        return point_ids

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_vector: list[float],
        top_k: int = 5,
        min_score: float = 0.3,
        task_id: str | None = None,
        scene_id: str | None = None,
        min_duration: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-K video segments most similar to `query_vector`.

        Optionally filter by task_id (restrict to clips indexed for this
        pipeline run) and scene_id (restrict to clips for a specific scene).

        Returns a list of dicts with keys:
          point_id, clip_id, video_id, local_path,
          start_seconds, end_seconds, score
        """
        self.ensure_collection()

        # Build optional payload filter
        must_conditions: list[qmodels.Condition] = []
        if task_id:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="task_id",
                    match=qmodels.MatchValue(value=task_id),
                )
            )
        if scene_id:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="scene_id",
                    match=qmodels.MatchValue(value=scene_id),
                )
            )
        if min_duration > 0:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="duration",
                    range=qmodels.Range(gte=min_duration),
                )
            )

        query_filter = (
            qmodels.Filter(must=must_conditions) if must_conditions else None
        )

        results = self._client.search(
            collection_name=COLLECTION,
            query_vector=(VECTOR_NAME, query_vector),
            query_filter=query_filter,
            limit=top_k,
            score_threshold=min_score,
            with_payload=True,
        )

        hits: list[dict[str, Any]] = []
        for r in results:
            payload = r.payload or {}
            hits.append(
                {
                    "point_id":      str(r.id),
                    "clip_id":       payload.get("clip_id", ""),
                    "video_id":      payload.get("video_id", ""),
                    "local_path":    payload.get("local_path", ""),
                    "start_seconds": payload.get("start_seconds", 0.0),
                    "end_seconds":   payload.get("end_seconds", 0.0),
                    "duration":      payload.get("duration", 0.0),
                    "scene_id":      payload.get("scene_id", ""),
                    "score":         r.score,
                }
            )

        logger.debug(
            "qdrant.query",
            hits=len(hits),
            top_score=hits[0]["score"] if hits else 0,
            scene_id=scene_id,
        )
        return hits

    # ── Batch query (for all scenes at once) ──────────────────────────────────

    def batch_query(
        self,
        queries: list[dict[str, Any]],
        top_k: int = 3,
        min_score: float = 0.3,
    ) -> list[list[dict[str, Any]]]:
        """
        Execute multiple vector queries in a single round-trip using
        Qdrant's batch search API.

        `queries` is a list of dicts:
          {"vector": [...], "task_id": "...", "scene_id": "..."}

        Returns a parallel list of hit lists.
        """
        self.ensure_collection()

        search_requests: list[qmodels.SearchRequest] = []
        for q in queries:
            must: list[qmodels.Condition] = []
            if q.get("task_id"):
                must.append(qmodels.FieldCondition(
                    key="task_id", match=qmodels.MatchValue(value=q["task_id"])
                ))
            if q.get("scene_id"):
                must.append(qmodels.FieldCondition(
                    key="scene_id", match=qmodels.MatchValue(value=q["scene_id"])
                ))

            search_requests.append(
                qmodels.SearchRequest(
                    vector=qmodels.NamedVector(name=VECTOR_NAME, vector=q["vector"]),
                    filter=qmodels.Filter(must=must) if must else None,
                    limit=top_k,
                    score_threshold=min_score,
                    with_payload=True,
                )
            )

        batch_results = self._client.search_batch(
            collection_name=COLLECTION,
            requests=search_requests,
        )

        all_hits: list[list[dict[str, Any]]] = []
        for result_set in batch_results:
            hits: list[dict[str, Any]] = []
            for r in result_set:
                payload = r.payload or {}
                hits.append({
                    "point_id":      str(r.id),
                    "clip_id":       payload.get("clip_id", ""),
                    "video_id":      payload.get("video_id", ""),
                    "local_path":    payload.get("local_path", ""),
                    "start_seconds": payload.get("start_seconds", 0.0),
                    "end_seconds":   payload.get("end_seconds", 0.0),
                    "duration":      payload.get("duration", 0.0),
                    "scene_id":      payload.get("scene_id", ""),
                    "score":         r.score,
                })
            all_hits.append(hits)

        return all_hits

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def delete_task_points(self, task_id: str) -> None:
        """Remove all points associated with a pipeline run."""
        try:
            self._client.delete(
                collection_name=COLLECTION,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="task_id",
                                match=qmodels.MatchValue(value=task_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            logger.info("qdrant.task_points_deleted", task_id=task_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("qdrant.delete_failed", task_id=task_id, error=str(exc))


# ── Singleton ─────────────────────────────────────────────────────────────────

_qdrant_service: QdrantService | None = None


def get_qdrant_service() -> QdrantService:
    global _qdrant_service  # noqa: PLW0603
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service
