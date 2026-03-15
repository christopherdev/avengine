"""
Pixabay stock footage sourcer.

Pixabay provides CC0-licensed video clips via a free search API.
Used as a secondary stock footage source after Pexels, before YouTube.

API docs: https://pixabay.com/api/docs/
Free tier: 100 requests/minute, no monthly cap.
"""
from __future__ import annotations

import pathlib
from typing import Any

import httpx
import structlog

from src.core.config import get_settings
from src.schemas.pipeline import SourcedClip

logger = structlog.get_logger(__name__)

_PIXABAY_VIDEO_URL = "https://pixabay.com/api/videos/"
_PREFER_QUALITIES = ("large", "medium", "small", "tiny")


async def search_and_download_pixabay(
    query: str,
    max_results: int = 3,
    clip_id_prefix: str = "",
    max_duration_seconds: int = 300,
) -> list[SourcedClip]:
    """
    Search Pixabay for stock footage matching `query` and download clips.
    Returns an empty list if PIXABAY_API_KEY is not configured.
    """
    settings = get_settings()
    if not settings.pixabay_api_key:
        return []

    scratch_dir = pathlib.Path(settings.video_scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    try:
        hits = await _search_pixabay(
            query=query,
            api_key=settings.pixabay_api_key,
            per_page=max_results * 2,
            max_duration=max_duration_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("pixabay.search_failed", query=query[:80], error=str(exc)[:200])
        return []

    clips: list[SourcedClip] = []
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        for i, hit in enumerate(hits):
            if len(clips) >= max_results:
                break
            file_info = _pick_best_file(hit.get("videos", {}))
            if not file_info or not file_info.get("url"):
                continue

            clip_id = f"{clip_id_prefix}pbx{i + 1:02d}"
            clip_dir = scratch_dir / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)
            dest = clip_dir / f"{clip_id}.mp4"

            try:
                await _stream_download(client, file_info["url"], dest)
                clips.append(
                    SourcedClip(
                        clip_id=clip_id,
                        source_url=f"https://pixabay.com/videos/id-{hit.get('id', '')}/",
                        local_path=str(dest),
                        platform="pixabay",
                        title=f"pixabay-{hit.get('id', '')}",
                        duration_seconds=float(hit.get("duration") or 0),
                        width=file_info.get("width"),
                        height=file_info.get("height"),
                        fps=None,
                        metadata={
                            "pixabay_id": hit.get("id"),
                            "tags": hit.get("tags", ""),
                            "user": hit.get("user", ""),
                        },
                    )
                )
                logger.info(
                    "pixabay.downloaded",
                    clip_id=clip_id,
                    pixabay_id=hit.get("id"),
                    duration=hit.get("duration"),
                    height=file_info.get("height"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "pixabay.download_failed",
                    clip_id=clip_id,
                    error=str(exc)[:200],
                )

    return clips


async def _search_pixabay(
    query: str,
    api_key: str,
    per_page: int = 6,
    max_duration: int = 300,
) -> list[dict[str, Any]]:
    import asyncio

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(3):
            resp = await client.get(
                _PIXABAY_VIDEO_URL,
                params={
                    "key": api_key,
                    "q": query,
                    "per_page": min(per_page, 20),
                    "video_type": "film",  # excludes animation
                    "safesearch": "true",
                },
            )
            if resp.status_code == 429:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "pixabay.rate_limited",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                )
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
            return [h for h in hits if (h.get("duration") or 0) <= max_duration]

    logger.warning("pixabay.rate_limit_exhausted", query=query[:60])
    return []


def _pick_best_file(videos: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the best quality MP4 available (large → medium → small → tiny)."""
    for quality in _PREFER_QUALITIES:
        entry = videos.get(quality)
        if entry and entry.get("url"):
            return entry
    return None


async def _stream_download(
    client: httpx.AsyncClient, url: str, dest: pathlib.Path
) -> None:
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                fh.write(chunk)
