"""
yt-dlp Extraction Module.

Responsibilities:
  - Accept a video URL + optional search query.
  - Rotate through the proxy pool to avoid rate-limiting and IP bans.
  - Extract full JSON metadata (info_dict) without downloading when possible
    (used by the Matching agent to verify clip suitability before committing
    bandwidth).
  - Download the best available video stream up to a configured resolution
    cap into the scratch directory.
  - Return a SourcedClip with all metadata attached.

Platform coverage via yt-dlp extractors:
  YouTube, Vimeo, Twitter/X, Reddit, TikTok (fallback to Playwright),
  Instagram (fallback to Playwright), Dailymotion, direct MP4 URLs.

Obfuscated platforms (TikTok, Instagram Reels) that require a logged-in
session or JavaScript rendering are caught here and re-routed to the
PlaywrightExtractor (see playwright_extractor.py).
"""
from __future__ import annotations

import asyncio
import pathlib
import uuid
from typing import Any

import structlog
import yt_dlp
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.agents.sourcing.proxy_pool import get_proxy_pool
from src.core.config import get_settings
from src.core.exceptions import DownloadError
from src.schemas.pipeline import SourcedClip

logger = structlog.get_logger(__name__)
settings = get_settings()

# Platforms that require the Playwright fallback
_PLAYWRIGHT_PLATFORMS = frozenset({"tiktok.com", "instagram.com", "reels"})

# Maximum video resolution to download (avoids pulling 4K files)
_MAX_HEIGHT = 1080

# yt-dlp format selector: best video+audio up to _MAX_HEIGHT, prefer mp4
_FORMAT_SELECTOR = (
    f"bestvideo[height<={_MAX_HEIGHT}][ext=mp4]+bestaudio[ext=m4a]"
    f"/bestvideo[height<={_MAX_HEIGHT}]+bestaudio"
    f"/best[height<={_MAX_HEIGHT}]"
    f"/best"
)


class YtDlpExtractor:
    """
    Wraps yt-dlp with proxy rotation, retry logic, and structured output.
    """

    def __init__(self) -> None:
        self._proxy_pool = get_proxy_pool()
        self._scratch_dir = pathlib.Path(settings.video_scratch_dir)
        self._scratch_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    async def extract_metadata(self, url: str) -> dict[str, Any]:
        """
        Fetch the yt-dlp info_dict for a URL without downloading.
        Used to validate clip suitability before committing bandwidth.
        """
        if self._needs_playwright(url):
            raise DownloadError(
                f"URL requires Playwright extraction: {url}",
                detail="route_to_playwright",
            )

        async with self._proxy_pool.checkout() as proxy_url:
            opts = self._base_opts(proxy_url, download=False)
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None, lambda: self._run_extract(url, opts)
            )
        return info

    async def download(self, url: str, clip_id: str | None = None) -> SourcedClip:
        """
        Download the video at `url` and return a populated SourcedClip.

        Retries up to 3 times, rotating the proxy on each attempt.
        Routes to PlaywrightExtractor automatically for obfuscated platforms.
        """
        if self._needs_playwright(url):
            raise DownloadError(
                f"URL requires Playwright extraction: {url}",
                detail="route_to_playwright",
            )

        clip_id = clip_id or str(uuid.uuid4())[:8]
        output_path = self._scratch_dir / clip_id / "%(title).80s.%(ext)s"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        last_exc: Exception | None = None

        for attempt in range(3):
            async with self._proxy_pool.checkout() as proxy_url:
                opts = self._base_opts(proxy_url, download=True)
                opts["outtmpl"] = str(output_path)

                try:
                    loop = asyncio.get_event_loop()
                    info = await loop.run_in_executor(
                        None, lambda: self._run_extract(url, opts)
                    )
                    local_path = self._find_downloaded_file(output_path.parent)
                    return self._build_sourced_clip(clip_id, url, local_path, info)

                except yt_dlp.utils.DownloadError as exc:
                    last_exc = exc
                    error_msg = str(exc).lower()
                    logger.warning(
                        "ytdlp.download_failed",
                        url=url[:80],
                        attempt=attempt + 1,
                        error=str(exc)[:200],
                    )
                    # Don't retry on these — they are permanent failures
                    if any(
                        phrase in error_msg
                        for phrase in ("video unavailable", "private video", "age-restricted")
                    ):
                        break

                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "ytdlp.unexpected_error",
                        url=url[:80],
                        attempt=attempt + 1,
                        error=str(exc)[:200],
                    )

        raise DownloadError(
            f"Failed to download {url} after 3 attempts.",
            detail=str(last_exc),
        )

    # ── yt-dlp helpers ─────────────────────────────────────────────────────────

    def _base_opts(self, proxy_url: str | None, *, download: bool) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "format": _FORMAT_SELECTOR,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": not download,
            "skip_download": not download,
            "noplaylist": True,
            "age_limit": None,
            # Write JSON sidecar alongside the video file
            "writeinfojson": download,
            # Merge video+audio into mp4
            "merge_output_format": "mp4",
            # Socket timeout
            "socket_timeout": 30,
            # Retry on transient network errors
            "retries": 2,
            "fragment_retries": 3,
            # Post-processor to embed thumbnails
            "writethumbnail": False,
        }

        if proxy_url:
            opts["proxy"] = proxy_url

        if settings.is_production:
            # Rate-limit downloads to be a good citizen
            opts["ratelimit"] = 2_000_000  # 2 MB/s

        return opts

    def _run_extract(self, url: str, opts: dict[str, Any]) -> dict[str, Any]:
        """
        Synchronous yt-dlp call.  Must be run in an executor to avoid
        blocking the event loop.
        """
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=opts.get("skip_download") is False)
            return ydl.sanitize_info(info or {})

    def _find_downloaded_file(self, directory: pathlib.Path) -> str:
        """Return the path of the most recently modified video file in directory."""
        video_extensions = {".mp4", ".webm", ".mkv", ".mov", ".avi"}
        candidates = [
            f for f in directory.iterdir()
            if f.suffix.lower() in video_extensions
        ]
        if not candidates:
            raise DownloadError(
                f"No video file found in {directory} after download.",
                detail="post_download_file_not_found",
            )
        return str(max(candidates, key=lambda f: f.stat().st_mtime))

    def _build_sourced_clip(
        self,
        clip_id: str,
        source_url: str,
        local_path: str,
        info: dict[str, Any],
    ) -> SourcedClip:
        return SourcedClip(
            clip_id=clip_id,
            source_url=source_url,
            local_path=local_path,
            platform=self._detect_platform(source_url),
            title=info.get("title", ""),
            duration_seconds=info.get("duration"),
            width=info.get("width"),
            height=info.get("height"),
            fps=info.get("fps"),
            metadata={
                "uploader": info.get("uploader", ""),
                "upload_date": info.get("upload_date", ""),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "description": (info.get("description") or "")[:500],
                "tags": info.get("tags", [])[:20],
                "categories": info.get("categories", []),
                "extractor": info.get("extractor", ""),
            },
        )

    @staticmethod
    def _needs_playwright(url: str) -> bool:
        url_lower = url.lower()
        return any(p in url_lower for p in _PLAYWRIGHT_PLATFORMS)

    @staticmethod
    def _detect_platform(url: str) -> str:
        url_lower = url.lower()
        for platform in ("youtube", "vimeo", "tiktok", "instagram", "twitter", "dailymotion"):
            if platform in url_lower:
                return platform
        return "unknown"


# ── Search-to-download convenience function ────────────────────────────────────

async def search_and_download(
    query: str,
    max_results: int = 3,
    clip_id_prefix: str = "",
    max_duration_seconds: int = 300,
) -> list[SourcedClip]:
    """
    Search YouTube for `query` and download up to `max_results` clips.

    Uses yt-dlp's ytsearch extractor to avoid the YouTube Data API quota.
    Fetches extra candidates to allow filtering out long videos — B-roll clips
    should be short (≤ max_duration_seconds, default 5 min).
    """
    extractor = YtDlpExtractor()
    # Fetch 2× candidates so we have spares after duration filtering
    fetch_n = max_results * 2
    search_url = f"ytsearch{fetch_n}:{query}"

    try:
        info = await extractor.extract_metadata(search_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ytdlp.search_failed", query=query[:80], error=str(exc)[:200])
        return []

    entries = info.get("entries") or []

    # Prefer short clips — filter by duration, fall back to anything if none pass
    short_entries = [
        e for e in entries
        if (e.get("duration") or 0) <= max_duration_seconds
    ]
    candidates = short_entries if short_entries else entries
    clips: list[SourcedClip] = []

    for i, entry in enumerate(candidates):
        if len(clips) >= max_results:
            break
        video_url = entry.get("webpage_url") or entry.get("url", "")
        if not video_url:
            continue

        clip_id = f"{clip_id_prefix}{i + 1:02d}"
        try:
            clip = await extractor.download(video_url, clip_id=clip_id)
            clips.append(clip)
            logger.info(
                "ytdlp.downloaded",
                clip_id=clip_id,
                title=clip.title[:60],
                duration=clip.duration_seconds,
            )
        except DownloadError as exc:
            logger.warning("ytdlp.skip", clip_id=clip_id, error=str(exc)[:200])

    return clips
