"""
Passive RSS/Atom Feed Monitor.

Responsibilities:
  - Periodically poll a configured list of RSS/Atom feeds.
  - Fetch and parse full article HTML with BeautifulSoup to extract
    embedded video URLs that feeds omit from their <enclosure> tags.
  - Deduplicate discovered URLs against a Redis Bloom-like seen-set
    (SADD + EXPIRE) so the same clip is never queued twice.
  - Publish new video URLs to the Redis `sourcing:discovered` list
    for the SourcingWorker to consume on-demand.

Scheduled via Celery beat every 5 minutes (see core/celery_app.py).
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any

import aiohttp
import feedparser
import structlog
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

# ── Feed registry ──────────────────────────────────────────────────────────────
# In production load this from a database table; here we seed from env/config.
DEFAULT_FEEDS: list[str] = [
    "https://www.youtube.com/feeds/videos.xml?channel_id=UCbmNph6atAoGfqLoCL_duAg",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://feeds.feedburner.com/TechCrunch",
    "https://www.theverge.com/rss/index.xml",
]

# Regex patterns that identify raw video URLs embedded in article HTML
_VIDEO_URL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r'https?://[^\s"\'<>]+\.mp4(?:\?[^\s"\'<>]*)?', re.IGNORECASE),
    re.compile(r'https?://[^\s"\'<>]+\.m3u8(?:\?[^\s"\'<>]*)?', re.IGNORECASE),
    re.compile(r'https?://[^\s"\'<>]+\.webm(?:\?[^\s"\'<>]*)?', re.IGNORECASE),
    # YouTube watch URLs
    re.compile(r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+', re.IGNORECASE),
    re.compile(r'https?://youtu\.be/[\w-]+', re.IGNORECASE),
    # Vimeo
    re.compile(r'https?://(?:www\.)?vimeo\.com/\d+', re.IGNORECASE),
]

_SEEN_KEY_PREFIX = "sourcing:seen:"
_QUEUE_KEY = "sourcing:discovered"
_SEEN_TTL = 604800  # 7 days


@dataclass
class DiscoveredVideo:
    url: str
    title: str
    source_feed: str
    platform: str
    entry_link: str


class RSSMonitor:
    """
    Async RSS monitor.  One instance per Celery beat invocation.
    """

    def __init__(self, feeds: list[str] | None = None) -> None:
        self._feeds = feeds or DEFAULT_FEEDS

    async def run(self) -> list[DiscoveredVideo]:
        """
        Poll all feeds, scrape articles, deduplicate, and enqueue new URLs.
        Returns the list of newly discovered videos this cycle.
        """
        import redis.asyncio as aioredis

        from src.core.config import get_settings

        settings = get_settings()
        redis = aioredis.from_url(str(settings.redis_url), decode_responses=True)

        connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=15, connect=5)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "AVEngine/0.1 RSS-Monitor (research bot)"},
        ) as session:
            tasks = [self._process_feed(feed_url, session, redis) for feed_url in self._feeds]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        all_discovered: list[DiscoveredVideo] = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("rss_monitor.feed_error", error=str(result))
            else:
                all_discovered.extend(result)

        await redis.aclose()

        logger.info(
            "rss_monitor.cycle_complete",
            feeds=len(self._feeds),
            discovered=len(all_discovered),
        )
        return all_discovered

    async def _process_feed(
        self,
        feed_url: str,
        session: aiohttp.ClientSession,
        redis: Any,
    ) -> list[DiscoveredVideo]:
        """Parse one feed and return new videos discovered from its entries."""
        try:
            feed_content = await self._fetch_text(session, feed_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rss_monitor.fetch_failed", url=feed_url, error=str(exc))
            return []

        parsed = feedparser.parse(feed_content)
        discovered: list[DiscoveredVideo] = []

        for entry in parsed.entries[:20]:  # cap at 20 entries per feed per cycle
            entry_url: str = getattr(entry, "link", "")
            if not entry_url:
                continue

            # Fast path: check enclosure tags for direct video links
            enclosure_videos = self._extract_enclosures(entry)

            # Slow path: scrape full article if no enclosures found
            if not enclosure_videos:
                try:
                    html = await self._fetch_text(session, entry_url)
                    enclosure_videos = self._extract_from_html(html, entry_url)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("rss_monitor.scrape_failed", url=entry_url, error=str(exc))

            for video_url in enclosure_videos:
                seen_key = _SEEN_KEY_PREFIX + _url_hash(video_url)
                if await redis.set(seen_key, "1", ex=_SEEN_TTL, nx=True):
                    # nx=True means SETNX — only succeeds if key didn't exist
                    video = DiscoveredVideo(
                        url=video_url,
                        title=getattr(entry, "title", ""),
                        source_feed=feed_url,
                        platform=_detect_platform(video_url),
                        entry_link=entry_url,
                    )
                    discovered.append(video)
                    # Push to the processing queue
                    await redis.rpush(_QUEUE_KEY, video_url)
                    logger.debug(
                        "rss_monitor.new_url",
                        url=video_url[:80],
                        platform=video.platform,
                    )

        return discovered

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    async def _fetch_text(self, session: aiohttp.ClientSession, url: str) -> str:
        async with session.get(url, allow_redirects=True) as resp:
            resp.raise_for_status()
            return await resp.text(errors="replace")

    def _extract_enclosures(self, entry: Any) -> list[str]:
        """Extract video URLs from RSS <enclosure> or <media:content> tags."""
        urls: list[str] = []
        for enc in getattr(entry, "enclosures", []):
            href = enc.get("href") or enc.get("url", "")
            mime = enc.get("type", "")
            if href and ("video" in mime or _is_video_url(href)):
                urls.append(href)

        for media in getattr(entry, "media_content", []):
            url = media.get("url", "")
            if url and _is_video_url(url):
                urls.append(url)

        return urls

    def _extract_from_html(self, html: str, base_url: str) -> list[str]:
        """
        Parse article HTML with BeautifulSoup to find embedded video URLs.

        Checks:
          1. <video src="..."> and <source src="..."> tags
          2. <iframe src="..."> YouTube/Vimeo embeds
          3. JSON-LD schema.org VideoObject
          4. Regex sweep over the raw HTML for video URL patterns
        """
        soup = BeautifulSoup(html, "lxml")
        urls: set[str] = set()

        # 1. Native video tags
        for tag in soup.find_all(["video", "source"]):
            src = tag.get("src") or tag.get("data-src") or ""
            if src and _is_video_url(src):
                urls.add(_make_absolute(src, base_url))

        # 2. iframes (YouTube / Vimeo embeds)
        for iframe in soup.find_all("iframe"):
            src = iframe.get("src", "")
            if "youtube.com/embed/" in src:
                video_id = src.split("/embed/")[-1].split("?")[0]
                urls.add(f"https://www.youtube.com/watch?v={video_id}")
            elif "player.vimeo.com/video/" in src:
                video_id = src.split("/video/")[-1].split("?")[0]
                urls.add(f"https://vimeo.com/{video_id}")

        # 3. JSON-LD VideoObject
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                import json

                data = json.loads(script.string or "{}")
                if isinstance(data, list):
                    data = data[0]
                if data.get("@type") == "VideoObject":
                    content_url = data.get("contentUrl") or data.get("embedUrl", "")
                    if content_url:
                        urls.add(content_url)
            except Exception:  # noqa: BLE001
                pass

        # 4. Regex sweep on raw HTML (catches obfuscated players)
        for pattern in _VIDEO_URL_PATTERNS:
            for match in pattern.findall(html):
                urls.add(match)

        return list(urls)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _is_video_url(url: str) -> bool:
    return bool(re.search(r'\.(mp4|m3u8|webm|mov|avi)(\?.*)?$', url, re.IGNORECASE))


def _detect_platform(url: str) -> str:
    url_lower = url.lower()
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    if "tiktok.com" in url_lower:
        return "tiktok"
    if "instagram.com" in url_lower:
        return "instagram"
    if "vimeo.com" in url_lower:
        return "vimeo"
    if "twitter.com" in url_lower or "x.com" in url_lower:
        return "twitter"
    if _is_video_url(url):
        return "direct"
    return "unknown"


def _make_absolute(url: str, base: str) -> str:
    if url.startswith("http"):
        return url
    from urllib.parse import urljoin

    return urljoin(base, url)
