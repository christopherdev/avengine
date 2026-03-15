"""
Pexels stock footage sourcer.

Pexels provides CC0-licensed short video clips via a simple search API.
These are purpose-built B-roll and far more likely to be topically relevant
than general YouTube search results.

API docs: https://www.pexels.com/api/documentation/
Free tier: 200 requests/hour, 20,000/month.
"""
from __future__ import annotations

import pathlib
from typing import Any

import httpx
import structlog

from src.core.config import get_settings
from src.schemas.pipeline import SourcedClip

logger = structlog.get_logger(__name__)

_PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
_PREFER_HEIGHTS = (1080, 720, 480)


async def search_and_download_pexels(
    query: str,
    max_results: int = 3,
    clip_id_prefix: str = "",
    max_duration_seconds: int = 300,
) -> list[SourcedClip]:
    """
    Search Pexels for stock footage matching `query` and download clips.
    Returns an empty list if PEXELS_API_KEY is not configured.
    """
    settings = get_settings()
    if not settings.pexels_api_key:
        return []

    scratch_dir = pathlib.Path(settings.video_scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    try:
        videos = await _search_pexels(
            query=query,
            api_key=settings.pexels_api_key,
            per_page=max_results * 2,
            max_duration=max_duration_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("pexels.search_failed", query=query[:80], error=str(exc)[:200])
        return []

    clips: list[SourcedClip] = []
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        for i, video in enumerate(videos):
            if len(clips) >= max_results:
                break
            file_info = _pick_best_file(video.get("video_files", []))
            if not file_info:
                continue

            clip_id = f"{clip_id_prefix}px{i + 1:02d}"
            clip_dir = scratch_dir / clip_id
            clip_dir.mkdir(parents=True, exist_ok=True)
            dest = clip_dir / f"{clip_id}.mp4"

            try:
                await _stream_download(client, file_info["link"], dest)
                clips.append(
                    SourcedClip(
                        clip_id=clip_id,
                        source_url=video.get("url", ""),
                        local_path=str(dest),
                        platform="pexels",
                        title=f"pexels-{video.get('id', '')}",
                        duration_seconds=float(video.get("duration") or 0),
                        width=file_info.get("width"),
                        height=file_info.get("height"),
                        fps=file_info.get("fps"),
                        metadata={
                            "pexels_id": video.get("id"),
                            "photographer": video.get("user", {}).get("name", ""),
                        },
                    )
                )
                logger.info(
                    "pexels.downloaded",
                    clip_id=clip_id,
                    pexels_id=video.get("id"),
                    duration=video.get("duration"),
                    height=file_info.get("height"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "pexels.download_failed",
                    clip_id=clip_id,
                    error=str(exc)[:200],
                )

    return clips


async def _search_pexels(
    query: str,
    api_key: str,
    per_page: int = 6,
    max_duration: int = 300,
) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            _PEXELS_SEARCH_URL,
            headers={"Authorization": api_key},
            params={
                "query": query,
                "per_page": per_page,
                "orientation": "landscape",
                "size": "medium",
            },
        )
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
    return [v for v in videos if (v.get("duration") or 0) <= max_duration]


def _pick_best_file(files: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the best MP4 by preferred resolution (1080p → 720p → 480p → any)."""
    mp4_files = [
        f for f in files
        if f.get("file_type") == "video/mp4" and f.get("link")
    ]
    if not mp4_files:
        return None
    for preferred_h in _PREFER_HEIGHTS:
        for f in mp4_files:
            if f.get("height") == preferred_h:
                return f
    return mp4_files[0]


async def _stream_download(
    client: httpx.AsyncClient, url: str, dest: pathlib.Path
) -> None:
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            async for chunk in resp.aiter_bytes(chunk_size=65536):
                fh.write(chunk)
