"""
Central settings module.

All configuration is read from environment variables (12-factor).
Secrets are never hard-coded; they are sourced from AWS Secrets Manager
or injected as environment variables by the container orchestrator.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Annotated

from pydantic import AnyHttpUrl, Field, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"


class LogLevel(str, Enum):
    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"
    critical = "CRITICAL"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env",),  # Only load local .env if it exists; prod/systemd uses injected env vars
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_name: str = "AVEngine"
    environment: Environment = Environment.development
    log_level: LogLevel = LogLevel.info
    debug: bool = False

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"  # noqa: S104
    api_port: int = 20000
    api_workers: int = 4
    api_prefix: str = "/api/v1"

    # ── Security ──────────────────────────────────────────────────────────────
    secret_key: str = Field(..., min_length=32)
    # Comma-separated list of valid API keys.  Empty string = auth disabled (dev only).
    # Production: API_KEYS=key1,key2
    api_keys: str = Field(default="", repr=False)
    # Explicit opt-in for unauthenticated access in development.
    # When False (default), requests without a valid key/JWT are rejected even in dev.
    allow_anonymous_dev: bool = False
    cors_origins: list[AnyHttpUrl] = []
    allowed_hosts: list[str] = ["*"]

    @property
    def api_key_list(self) -> list[str]:
        """Parsed list of valid API keys (empty list = auth disabled)."""
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    # ── JWT ───────────────────────────────────────────────────────────────────
    # Falls back to secret_key when not explicitly set.
    jwt_secret_key: str = Field(default="", repr=False)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours

    @property
    def effective_jwt_secret(self) -> str:
        return self.jwt_secret_key or self.secret_key

    # ── Bootstrap admin ───────────────────────────────────────────────────────
    # Created automatically on first startup if no users exist.
    # ADMIN_PASSWORD has no default — it MUST be set via environment variable.
    # This prevents silent deployment with a well-known password.
    admin_username: str = "admin"
    admin_password: str = Field(..., min_length=8, repr=False)

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://avengine:avengine@localhost:5432/avengine"
    )
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_timeout: int = 30

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")  # type: ignore[assignment]
    redis_max_connections: int = 50

    # ── Celery ────────────────────────────────────────────────────────────────
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_time_limit: int = 3600  # 1 hour hard limit
    celery_task_soft_time_limit: int = 3300

    # ── LLM Providers ─────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", repr=False)
    anthropic_api_key: str = Field(default="", repr=False)
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-6"

    # ── Serper (used by CrewAI SerperDevTool via os.environ) ──────────────────
    serper_api_key: str = Field(default="", repr=False)

    # ── Pexels (free stock footage API — primary B-roll source) ──────────────
    pexels_api_key: str = Field(default="", repr=False)

    # ── Pixabay (secondary stock footage API — fallback after Pexels) ─────────
    pixabay_api_key: str = Field(default="", repr=False)

    # ── TwelveLabs ────────────────────────────────────────────────────────────
    twelvelabs_api_key: str = Field(default="", repr=False)
    twelvelabs_index_id: str = ""

    # ── ElevenLabs ────────────────────────────────────────────────────────────
    elevenlabs_api_key: str = Field(default="", repr=False)
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel

    # ── AWS ───────────────────────────────────────────────────────────────────
    aws_region: str = "us-east-1"
    aws_s3_bucket: str = ""
    aws_access_key_id: str = Field(default="", repr=False)
    aws_secret_access_key: str = Field(default="", repr=False)

    # ── Proxy (for yt-dlp / Playwright) ───────────────────────────────────────
    proxy_endpoint: str = ""
    proxy_username: str = Field(default="", repr=False)
    proxy_password: str = Field(default="", repr=False)

    # ── Vector DB (Qdrant) ────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = Field(default="", repr=False)
    qdrant_collection_name: str = "video_embeddings"

    # ── Video Processing ──────────────────────────────────────────────────────
    video_output_dir: str = "/tmp/avengine/output"  # noqa: S108
    video_scratch_dir: str = "/tmp/avengine/scratch"  # noqa: S108
    video_max_duration_seconds: int = 300
    ffmpeg_threads: int = 4

    # ── Observability ─────────────────────────────────────────────────────────
    sentry_dsn: str = ""
    otlp_endpoint: str = "http://localhost:4317"
    enable_tracing: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_csv(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def proxy_url(self) -> str | None:
        if self.proxy_endpoint and self.proxy_username:
            return (
                f"http://{self.proxy_username}:{self.proxy_password}@{self.proxy_endpoint}"
            )
        return None

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.production


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()  # type: ignore[call-arg]
