"""
Playwright CDN Interceptor.

Handles platforms with obfuscated video players that yt-dlp cannot reach
without a running browser (TikTok, Instagram Reels, some news-site players).

Strategy:
  1. Launch a headless Chromium context routed through a residential proxy.
  2. Register a network request interception handler before navigation.
  3. Navigate to the target URL and wait for video network activity.
  4. Intercept:
       - Direct .mp4 / .webm / .mov CDN requests
       - HLS manifest (.m3u8) requests
       - Blob object URL creation (caught via JS injection)
       - Signed CDN tokens in XHR/Fetch responses (TikTok pattern)
  5. Download the first qualifying CDN URL with aiohttp + proxy.
  6. Return a SourcedClip.

TikTok-specific flow:
  TikTok serves video via a signed CDN URL embedded in a JSON API response.
  We intercept the `POST /api/item/detail/` XHR, parse the JSON, and extract
  the `video.playAddr` field — this is the raw MP4 CDN URL.

Instagram Reels-specific flow:
  Instagram serves Reels via a GraphQL endpoint.  We intercept
  `POST /graphql` and extract `video_url` from the response JSON.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import re
import uuid
from typing import Any
from urllib.parse import urlparse

import aiohttp
import structlog

from src.agents.sourcing.proxy_pool import get_proxy_pool
from src.core.config import get_settings
from src.core.exceptions import DownloadError
from src.schemas.pipeline import SourcedClip

logger = structlog.get_logger(__name__)
settings = get_settings()

# CDN URL patterns to intercept
_CDN_VIDEO_PATTERN = re.compile(
    r'https?://[^\s"\'<>]+'
    r'(?:\.mp4|\.webm|\.mov|\.m3u8|/video/tos/|/videoplayback\?|/media/|muscdn\.com)',
    re.IGNORECASE,
)

_TIKTOK_API_PATTERN = re.compile(r'/api/item/detail|/aweme/v1/feed', re.IGNORECASE)
_INSTAGRAM_GQL_PATTERN = re.compile(r'/graphql/query', re.IGNORECASE)

# JS to expose blob URLs created by the player
_BLOB_INTERCEPT_JS = """
(function() {
    const _createObjectURL = URL.createObjectURL.bind(URL);
    URL.createObjectURL = function(blob) {
        const url = _createObjectURL(blob);
        window.__blobUrls = window.__blobUrls || [];
        window.__blobUrls.push({url: url, size: blob.size, type: blob.type});
        return url;
    };
})();
"""


class PlaywrightExtractor:
    """
    Headless browser extractor for obfuscated video platforms.

    One instance is used per extraction — each call creates a fresh
    browser context to avoid cookie bleed between requests.
    """

    def __init__(self) -> None:
        self._proxy_pool = get_proxy_pool()
        self._scratch_dir = pathlib.Path(settings.video_scratch_dir)
        self._scratch_dir.mkdir(parents=True, exist_ok=True)

    async def extract(self, url: str, clip_id: str | None = None) -> SourcedClip:
        """
        Navigate to `url` in a headless Chromium browser, intercept the
        video CDN request, download the file, and return a SourcedClip.
        """
        from playwright.async_api import async_playwright

        clip_id = clip_id or str(uuid.uuid4())[:8]
        platform = self._detect_platform(url)

        logger.info("playwright.starting", url=url[:80], platform=platform, clip_id=clip_id)

        async with self._proxy_pool.checkout() as proxy_url:
            proxy_config = self._build_proxy_config(proxy_url)

            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=True,
                    proxy=proxy_config,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )

                context = await browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                    locale="en-US",
                    timezone_id="America/New_York",
                    java_script_enabled=True,
                    bypass_csp=True,
                    extra_http_headers={
                        "Accept-Language": "en-US,en;q=0.9",
                        "DNT": "1",
                    },
                )

                # Inject blob-intercept script before any page code runs
                await context.add_init_script(script=_BLOB_INTERCEPT_JS)

                # Shared mutable accumulator for intercepted CDN URLs
                intercepted: list[dict[str, Any]] = []

                page = await context.new_page()

                # Register request + response intercept handlers
                page.on("request", lambda req: self._on_request(req, intercepted))
                page.on("response", self._make_response_handler(intercepted, platform))

                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                    # Wait for video player to initialise and fire network requests
                    await self._wait_for_video(page, intercepted)

                    # Check for blob URLs injected via JS
                    blob_urls = await page.evaluate("window.__blobUrls || []")
                    if blob_urls:
                        logger.debug("playwright.blob_urls", count=len(blob_urls))

                finally:
                    await context.close()
                    await browser.close()

        if not intercepted:
            raise DownloadError(
                f"No CDN video URL intercepted for {url}",
                detail="playwright_no_cdn_url",
            )

        # Pick the best candidate (prefer mp4 over m3u8; prefer largest)
        cdn_url = self._pick_best_cdn(intercepted)
        logger.info("playwright.cdn_url_selected", url=cdn_url[:80], clip_id=clip_id)

        # Download the CDN URL with aiohttp
        local_path = await self._download_cdn(cdn_url, clip_id, proxy_url=proxy_url)

        return SourcedClip(
            clip_id=clip_id,
            source_url=url,
            local_path=local_path,
            platform=platform,
            title="",
            metadata={
                "cdn_url": cdn_url,
                "intercepted_count": len(intercepted),
                "extraction_method": "playwright",
            },
        )

    # ── Intercept handlers ─────────────────────────────────────────────────────

    def _on_request(self, request: Any, intercepted: list) -> None:
        """Capture direct CDN requests (e.g. plain MP4 GET)."""
        url = request.url
        if _CDN_VIDEO_PATTERN.search(url):
            intercepted.append({"url": url, "type": "request", "format": self._url_format(url)})
            logger.debug("playwright.request_intercepted", url=url[:80])

    def _make_response_handler(self, intercepted: list, platform: str) -> Any:
        """
        Return an async response handler closure.

        Parses JSON API responses from TikTok and Instagram to extract
        the signed CDN video URL.
        """
        async def _on_response(response: Any) -> None:
            url = response.url
            status = response.status

            if status != 200:
                return

            # TikTok: parse item detail API response
            if platform == "tiktok" and _TIKTOK_API_PATTERN.search(url):
                try:
                    body = await response.json()
                    cdn = self._extract_tiktok_cdn(body)
                    if cdn:
                        intercepted.append({"url": cdn, "type": "api_json", "format": "mp4"})
                        logger.debug("playwright.tiktok_cdn", url=cdn[:80])
                except Exception:  # noqa: BLE001
                    pass

            # Instagram: parse GraphQL response
            elif platform == "instagram" and _INSTAGRAM_GQL_PATTERN.search(url):
                try:
                    body = await response.json()
                    cdn = self._extract_instagram_cdn(body)
                    if cdn:
                        intercepted.append({"url": cdn, "type": "api_json", "format": "mp4"})
                        logger.debug("playwright.instagram_cdn", url=cdn[:80])
                except Exception:  # noqa: BLE001
                    pass

            # Generic: check Content-Type for video streams
            elif "video" in response.headers.get("content-type", ""):
                if _CDN_VIDEO_PATTERN.search(url):
                    intercepted.append({"url": url, "type": "content_type", "format": self._url_format(url)})

        return _on_response

    # ── Platform-specific JSON parsers ─────────────────────────────────────────

    def _extract_tiktok_cdn(self, body: Any) -> str | None:
        """
        TikTok API item detail structure:
          body.itemInfo.itemStruct.video.playAddr
          body.aweme_list[0].video.play_addr.url_list[0]
        """
        try:
            # v1 structure
            item = body.get("itemInfo", {}).get("itemStruct", {})
            if item:
                addr = item.get("video", {}).get("playAddr", "")
                if addr:
                    return addr

            # v2 feed structure
            aweme_list = body.get("aweme_list", [])
            if aweme_list:
                url_list = (
                    aweme_list[0]
                    .get("video", {})
                    .get("play_addr", {})
                    .get("url_list", [])
                )
                if url_list:
                    return url_list[0]
        except (KeyError, IndexError, TypeError):
            pass
        return None

    def _extract_instagram_cdn(self, body: Any) -> str | None:
        """
        Instagram GraphQL structure varies by endpoint.
        Walk the dict tree looking for 'video_url'.
        """
        return self._deep_find(body, "video_url")

    def _deep_find(self, obj: Any, key: str, depth: int = 0) -> str | None:
        """Recursively search a JSON structure for a specific key."""
        if depth > 8:
            return None
        if isinstance(obj, dict):
            if key in obj and isinstance(obj[key], str) and obj[key].startswith("http"):
                return obj[key]
            for v in obj.values():
                result = self._deep_find(v, key, depth + 1)
                if result:
                    return result
        elif isinstance(obj, list):
            for item in obj[:5]:  # cap depth to avoid huge lists
                result = self._deep_find(item, key, depth + 1)
                if result:
                    return result
        return None

    # ── Waiting & selection helpers ────────────────────────────────────────────

    async def _wait_for_video(self, page: Any, intercepted: list) -> None:
        """
        Wait for either a video CDN URL to be intercepted or a timeout.
        Also tries clicking the play button to trigger lazy-load players.
        """
        for attempt in range(6):  # up to 6 × 2s = 12s
            if intercepted:
                return

            # Try clicking the play button on common selectors
            if attempt == 1:
                for selector in (
                    "button[aria-label*='play' i]",
                    ".play-button",
                    "video",
                    "[data-testid='play-button']",
                ):
                    try:
                        await page.click(selector, timeout=1000)
                        break
                    except Exception:  # noqa: BLE001
                        pass

            await asyncio.sleep(2)

    def _pick_best_cdn(self, intercepted: list[dict[str, Any]]) -> str:
        """
        Prefer: mp4 > webm > m3u8.
        Among equals, prefer the URL that looks like a CDN (no query string
        complexity = more stable).
        """
        def _score(entry: dict) -> tuple[int, int]:
            fmt = entry.get("format", "")
            fmt_score = {"mp4": 3, "webm": 2, "mov": 2, "m3u8": 1}.get(fmt, 0)
            # Prefer shorter URLs (often the primary CDN, not a redirect)
            url_score = -len(entry["url"])
            return (fmt_score, url_score)

        return max(intercepted, key=_score)["url"]

    # ── Download ───────────────────────────────────────────────────────────────

    async def _download_cdn(
        self, cdn_url: str, clip_id: str, *, proxy_url: str | None
    ) -> str:
        """Stream-download a CDN URL to the scratch directory."""
        output_dir = self._scratch_dir / clip_id
        output_dir.mkdir(parents=True, exist_ok=True)

        ext = self._url_format(cdn_url) or "mp4"
        dest = output_dir / f"clip.{ext}"

        connector = aiohttp.TCPConnector()
        timeout = aiohttp.ClientTimeout(total=120, connect=10)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.tiktok.com/",
        }

        proxy = proxy_url  # aiohttp accepts http:// proxy strings directly

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=headers
        ) as session:
            async with session.get(cdn_url, proxy=proxy, allow_redirects=True) as resp:
                resp.raise_for_status()
                with dest.open("wb") as fh:
                    async for chunk in resp.content.iter_chunked(65536):
                        fh.write(chunk)

        logger.info(
            "playwright.download_complete",
            clip_id=clip_id,
            bytes=dest.stat().st_size,
            path=str(dest),
        )
        return str(dest)

    # ── Static helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _detect_platform(url: str) -> str:
        url_lower = url.lower()
        if "tiktok.com" in url_lower:
            return "tiktok"
        if "instagram.com" in url_lower:
            return "instagram"
        return "unknown"

    @staticmethod
    def _url_format(url: str) -> str:
        path = urlparse(url).path.lower()
        for ext in ("mp4", "webm", "mov", "m3u8", "mkv"):
            if path.endswith(f".{ext}"):
                return ext
        return "mp4"

    @staticmethod
    def _build_proxy_config(proxy_url: str | None) -> dict | None:
        if not proxy_url:
            return None
        parsed = urlparse(proxy_url)
        config: dict[str, Any] = {
            "server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
        }
        if parsed.username:
            config["username"] = parsed.username
            config["password"] = parsed.password or ""
        return config
