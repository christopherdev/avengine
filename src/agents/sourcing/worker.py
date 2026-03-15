"""
Sourcing Worker — orchestrates all sourcing strategies per pipeline run.

For each B-roll cue in the VideoScript, the worker:

  1. Builds a prioritised list of candidate URLs:
       a) Seed URLs passed in the API request (highest priority)
       b) URLs queued by the RSS monitor in Redis
       c) Fresh YouTube search using the cue's search_query

  2. For each candidate URL, routes to the correct extractor:
       - yt-dlp  → YouTube, Vimeo, Twitter, direct MP4, most platforms
       - Playwright → TikTok, Instagram Reels (JS-rendered players)

  3. Deduplicates clips by source_url hash.
  4. Caps total clips per cue at MAX_CLIPS_PER_CUE to bound disk usage.
  5. Returns a flat list of SourcedClip objects for the Matching agent.

The worker runs synchronously from the LangGraph node (via `run_sync`)
using asyncio.run() to drive its internal async methods.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import structlog

from src.agents.sourcing.playwright_extractor import PlaywrightExtractor
from src.agents.sourcing.ytdlp_extractor import YtDlpExtractor, search_and_download
from src.core.config import get_settings
from src.core.exceptions import DownloadError, SourcingError
from src.schemas.pipeline import BRollCue, ScriptScene, SourcedClip

logger = structlog.get_logger(__name__)
settings = get_settings()

MAX_CLIPS_PER_CUE = 3
MAX_CONCURRENT_DOWNLOADS = 4

# Platforms routed to Playwright
_PLAYWRIGHT_PLATFORMS = frozenset({"tiktok.com", "instagram.com"})


class SourcingWorker:
    """
    Orchestrates all sourcing strategies for a pipeline run.
    """

    def __init__(self) -> None:
        self._ytdlp = YtDlpExtractor()
        self._playwright = PlaywrightExtractor()

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_sync(
        self,
        script_dict: dict[str, Any],
        seed_urls: list[str],
        task_id: str,
        topic: str = "",
    ) -> list[SourcedClip]:
        """
        Synchronous entry point called from the LangGraph node.
        Drives the internal async workflow via asyncio.run().
        """
        try:
            return asyncio.run(
                self._run_async(script_dict, seed_urls, task_id, topic=topic)
            )
        except Exception as exc:  # noqa: BLE001
            raise SourcingError(
                f"Sourcing failed for task {task_id}: {exc}",
                detail=str(exc),
            ) from exc

    async def _run_async(
        self,
        script_dict: dict[str, Any],
        seed_urls: list[str],
        task_id: str,
        topic: str = "",
    ) -> list[SourcedClip]:
        """
        Main async sourcing loop.
        """
        from src.pipeline.publisher import async_publish
        from src.schemas.api import AgentStage

        scenes: list[dict] = script_dict.get("scenes", [])
        all_clips: list[SourcedClip] = []
        seen_hashes: set[str] = set()

        # Flatten all B-roll cues from all scenes
        cues: list[tuple[str, BRollCue]] = []
        for scene_raw in scenes:
            scene_id = scene_raw.get("scene_id", "")
            for cue_raw in scene_raw.get("b_roll_cues", []):
                try:
                    cue = BRollCue(**cue_raw)
                    cues.append((scene_id, cue))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("sourcing.bad_cue", error=str(exc))

        if not cues:
            logger.warning("sourcing.no_cues", task_id=task_id)
            return []

        # Publish initial progress
        total_cues = len(cues)
        await async_publish(
            task_id, AgentStage.sourcing,
            f"Sourcing footage for {total_cues} B-roll cues...",
            progress=35.0,
        )

        # Seed URL pool — shared across all cues (prioritised first)
        seed_pool = list(seed_urls)
        queued_urls = await self._drain_redis_queue(max_items=20)
        seed_pool.extend(queued_urls)

        # Process cues with bounded concurrency
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

        async def _process_cue(idx: int, scene_id: str, cue: BRollCue) -> list[SourcedClip]:
            async with semaphore:
                return await self._source_cue(
                    cue=cue,
                    scene_id=scene_id,
                    clip_id_prefix=f"{scene_id}{cue.cue_id}",
                    seed_pool=seed_pool,
                    seen_hashes=seen_hashes,
                    topic=topic,
                )

        tasks = [
            _process_cue(i, scene_id, cue)
            for i, (scene_id, cue) in enumerate(cues)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning("sourcing.cue_failed", error=str(result)[:200])
            else:
                all_clips.extend(result)

        await async_publish(
            task_id, AgentStage.sourcing,
            f"Sourced {len(all_clips)} clips from {total_cues} cues.",
            progress=54.0,
            data={"clip_count": len(all_clips)},
        )

        logger.info(
            "sourcing.complete",
            task_id=task_id,
            clips=len(all_clips),
            cues=total_cues,
        )
        return all_clips

    # ── Per-cue sourcing logic ─────────────────────────────────────────────────

    async def _source_cue(
        self,
        cue: BRollCue,
        scene_id: str,
        clip_id_prefix: str,
        seed_pool: list[str],
        seen_hashes: set[str],
        topic: str = "",
    ) -> list[SourcedClip]:
        """
        Source clips for a single B-roll cue.

        Priority order:
          1. Seed URLs that match the cue's keywords
          2. YouTube search for cue.search_query
          3. Playwright fallback for any URL that failed yt-dlp
        """
        clips: list[SourcedClip] = []
        playwright_fallbacks: list[str] = []

        # 1. Try relevant seed URLs first
        relevant_seeds = self._filter_seeds(seed_pool, cue.search_query)
        for seed_url in relevant_seeds[:2]:
            if len(clips) >= MAX_CLIPS_PER_CUE:
                break
            clip = await self._download_single(
                url=seed_url,
                clip_id=f"{clip_id_prefix}s{len(clips):02d}",
                seen_hashes=seen_hashes,
                playwright_fallbacks=playwright_fallbacks,
            )
            if clip:
                clips.append(clip)

        # Build topic-scoped search query (used by both Pexels and YouTube)
        search_query = f"{topic} {cue.search_query}".strip() if topic else cue.search_query

        # 2. Pexels stock footage (primary — purpose-built B-roll, much more relevant)
        if len(clips) < MAX_CLIPS_PER_CUE:
            need = MAX_CLIPS_PER_CUE - len(clips)
            try:
                from src.agents.sourcing.pexels_sourcer import search_and_download_pexels
                pexels_clips = await search_and_download_pexels(
                    query=search_query,
                    max_results=need,
                    clip_id_prefix=f"{clip_id_prefix}px",
                )
                for sc in pexels_clips:
                    url_hash = _hash(sc.source_url)
                    if url_hash not in seen_hashes:
                        seen_hashes.add(url_hash)
                        clips.append(sc)
                if pexels_clips:
                    logger.debug(
                        "sourcing.pexels_ok",
                        cue_id=cue.cue_id,
                        clips=len(pexels_clips),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "sourcing.pexels_failed",
                    query=search_query[:60],
                    error=str(exc)[:200],
                )

        # 3. Pixabay stock footage (secondary — larger library, fills Pexels gaps)
        if len(clips) < MAX_CLIPS_PER_CUE:
            need = MAX_CLIPS_PER_CUE - len(clips)
            try:
                from src.agents.sourcing.pixabay_sourcer import search_and_download_pixabay
                pixabay_clips = await search_and_download_pixabay(
                    query=search_query,
                    max_results=need,
                    clip_id_prefix=f"{clip_id_prefix}pbx",
                )
                for sc in pixabay_clips:
                    url_hash = _hash(sc.source_url)
                    if url_hash not in seen_hashes:
                        seen_hashes.add(url_hash)
                        clips.append(sc)
                if pixabay_clips:
                    logger.debug(
                        "sourcing.pixabay_ok",
                        cue_id=cue.cue_id,
                        clips=len(pixabay_clips),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "sourcing.pixabay_failed",
                    query=search_query[:60],
                    error=str(exc)[:200],
                )

        # 4. YouTube search (last resort — used when stock footage APIs have no results)
        if len(clips) < MAX_CLIPS_PER_CUE:
            need = MAX_CLIPS_PER_CUE - len(clips)
            try:
                search_clips = await search_and_download(
                    query=search_query,
                    max_results=need,
                    clip_id_prefix=f"{clip_id_prefix}y",
                )
                for sc in search_clips:
                    url_hash = _hash(sc.source_url)
                    if url_hash not in seen_hashes:
                        seen_hashes.add(url_hash)
                        clips.append(sc)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "sourcing.search_failed",
                    query=search_query[:60],
                    error=str(exc)[:200],
                )

        # 5. Playwright fallback for any URLs that yt-dlp rejected
        for url in playwright_fallbacks:
            if len(clips) >= MAX_CLIPS_PER_CUE:
                break
            clip = await self._playwright_download(
                url=url,
                clip_id=f"{clip_id_prefix}p{len(clips):02d}",
                seen_hashes=seen_hashes,
            )
            if clip:
                clips.append(clip)

        logger.debug(
            "sourcing.cue_done",
            cue_id=cue.cue_id,
            clips=len(clips),
            query=cue.search_query[:60],
        )
        return clips

    # ── Download helpers ───────────────────────────────────────────────────────

    async def _download_single(
        self,
        url: str,
        clip_id: str,
        seen_hashes: set[str],
        playwright_fallbacks: list[str],
    ) -> SourcedClip | None:
        url_hash = _hash(url)
        if url_hash in seen_hashes:
            return None
        seen_hashes.add(url_hash)

        needs_playwright = any(p in url.lower() for p in _PLAYWRIGHT_PLATFORMS)
        if needs_playwright:
            playwright_fallbacks.append(url)
            return None

        try:
            clip = await self._ytdlp.download(url, clip_id=clip_id)
            logger.debug("sourcing.ytdlp_ok", clip_id=clip_id, url=url[:60])
            return clip
        except DownloadError as exc:
            if "route_to_playwright" in str(exc.detail or ""):
                playwright_fallbacks.append(url)
            else:
                logger.debug("sourcing.ytdlp_skip", url=url[:60], reason=str(exc)[:100])
            return None

    async def _playwright_download(
        self,
        url: str,
        clip_id: str,
        seen_hashes: set[str],
    ) -> SourcedClip | None:
        try:
            clip = await self._playwright.extract(url, clip_id=clip_id)
            seen_hashes.add(_hash(url))
            logger.debug("sourcing.playwright_ok", clip_id=clip_id, url=url[:60])
            return clip
        except Exception as exc:  # noqa: BLE001
            logger.warning("sourcing.playwright_failed", url=url[:60], error=str(exc)[:200])
            return None

    # ── Redis queue drain ──────────────────────────────────────────────────────

    async def _drain_redis_queue(self, max_items: int = 20) -> list[str]:
        """
        Pop URLs from the `sourcing:discovered` queue that the RSS monitor
        populated.  Returns up to max_items URLs.
        """
        try:
            import redis.asyncio as aioredis

            redis = aioredis.from_url(str(settings.redis_url), decode_responses=True)
            urls: list[str] = []
            for _ in range(max_items):
                url = await redis.lpop("sourcing:discovered")
                if url is None:
                    break
                urls.append(url)
            await redis.aclose()
            return urls
        except Exception as exc:  # noqa: BLE001
            logger.warning("sourcing.redis_drain_failed", error=str(exc))
            return []

    # ── Seed filtering ─────────────────────────────────────────────────────────

    @staticmethod
    def _filter_seeds(seed_pool: list[str], query: str) -> list[str]:
        """
        Return seeds likely relevant to the query by checking keyword overlap.
        Falls back to returning all seeds if none match.
        """
        query_words = set(query.lower().split())
        scored = [
            (sum(1 for w in query_words if w in url.lower()), url)
            for url in seed_pool
        ]
        scored.sort(reverse=True)
        relevant = [url for score, url in scored if score > 0]
        return relevant or seed_pool


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]
