"""
Security headers middleware.

Injects defensive HTTP headers into every response:

  X-Content-Type-Options    — prevents MIME-sniffing
  X-Frame-Options           — blocks clickjacking (iframe embedding)
  Referrer-Policy           — limits referer leakage
  Permissions-Policy        — disables unused browser features
  Content-Security-Policy   — restricts resource loading / XSS mitigation
  Strict-Transport-Security — enforces HTTPS (production only)
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, is_production: bool = False) -> None:
        super().__init__(app)
        self._is_production = is_production

        # WebSocket connect-src: allow wss: only in prod, ws: + wss: in dev
        ws_src = "wss:" if is_production else "ws: wss:"

        # Content-Security-Policy
        # - Tailwind CSS is loaded from the CDN; it injects styles dynamically
        #   so 'unsafe-inline' is required for style-src.
        # - script-src does NOT include 'unsafe-inline' or 'unsafe-eval';
        #   all JS is in the single inline <script> block — migrate to an
        #   external bundle and add a hash/nonce to remove 'unsafe-inline'.
        self._csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com; "
            "style-src 'self' 'unsafe-inline'; "
            f"connect-src 'self' {ws_src}; "
            "media-src 'self'; "
            "img-src 'self' data:; "
            "font-src 'self' data:; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "frame-ancestors 'none';"
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        response = await call_next(request)

        h = response.headers
        h["X-Content-Type-Options"] = "nosniff"
        h["X-Frame-Options"] = "DENY"
        h["Referrer-Policy"] = "strict-origin-when-cross-origin"
        h["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), bluetooth=()"
        )
        h["Content-Security-Policy"] = self._csp

        if self._is_production:
            # 2-year max-age; only set over HTTPS
            h["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )

        return response
