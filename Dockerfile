# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
#  AVEngine — Multi-Stage Dockerfile
#  Optimised for AWS Fargate (arm64 / x86_64)
#
#  Stages:
#   1. base          — common OS deps + non-root user
#   2. builder       — Python deps compiled in an isolated layer
#   3. playwright    — install Chromium browsers (heavy, cached layer)
#   4. runtime       — minimal final image
# ─────────────────────────────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.11
ARG DEBIAN_CODENAME=bookworm

# ──────────────────────────────────────────── Stage 1: base ──────────────────
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_CODENAME} AS base

# Reproducible builds + sane defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# System packages needed at *runtime*
RUN apt-get update && apt-get install -y --no-install-recommends \
        # FFmpeg (video processing)
        ffmpeg \
        # libGL (MoviePy / OpenCV)
        libgl1 \
        libglib2.0-0 \
        # TLS / certs
        ca-certificates \
        # Playwright system deps (minimal set; full set added in playwright stage)
        libgbm1 \
        libasound2 \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────────────────────
# Use a specific UID/GID for consistent permissions across environments
ENV USER_ID=10001 \
    GROUP_ID=10001 \
    USER_NAME=avengine

RUN groupadd --gid ${GROUP_ID} ${USER_NAME} \
    && useradd --uid ${USER_ID} --gid ${GROUP_ID} --create-home --shell /usr/sbin/nologin ${USER_NAME}

WORKDIR /app

# ──────────────────────────────────────────── Stage 2: builder ───────────────
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast resolver / installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy only dependency manifests first — maximises layer cache reuse
COPY pyproject.toml ./
# Synthesise a minimal stub package so uv can resolve without the full source
RUN mkdir -p src && touch src/__init__.py

# Install all dependencies into an isolated venv
ENV VIRTUAL_ENV=/build/.venv
RUN uv venv $VIRTUAL_ENV \
    && uv pip install --no-cache --python $VIRTUAL_ENV/bin/python \
        -e ".[dev]" \
    || uv pip install --no-cache --python $VIRTUAL_ENV/bin/python \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.30.0" \
        "httpx>=0.27.0" \
        "websockets>=13.0" \
        "python-multipart>=0.0.9" \
        "celery[redis]>=5.4.0" \
        "redis>=5.0.0" \
        "aioredis>=2.0.1" \
        "langgraph>=0.2.0" \
        "langchain>=0.3.0" \
        "langchain-openai>=0.2.0" \
        "langchain-anthropic>=0.2.0" \
        "crewai>=0.80.0" \
        "crewai-tools>=0.14.0" \
        "pydantic-ai>=0.0.13" \
        "pydantic>=2.9.0" \
        "pydantic-settings>=2.5.0" \
        "moviepy>=2.0.0" \
        "yt-dlp>=2024.11.0" \
        "ffmpeg-python>=0.2.0" \
        "playwright>=1.48.0" \
        "beautifulsoup4>=4.12.0" \
        "feedparser>=6.0.11" \
        "lxml>=5.3.0" \
        "aiohttp>=3.10.0" \
        "twelvelabs>=0.3.0" \
        "qdrant-client>=1.12.0" \
        "lancedb>=0.13.0" \
        "elevenlabs>=1.9.0" \
        "opentelemetry-api>=1.27.0" \
        "opentelemetry-sdk>=1.27.0" \
        "opentelemetry-instrumentation-fastapi>=0.48b0" \
        "opentelemetry-exporter-otlp>=1.27.0" \
        "structlog>=24.4.0" \
        "sentry-sdk[fastapi]>=2.16.0" \
        "sqlalchemy[asyncio]>=2.0.36" \
        "alembic>=1.13.0" \
        "asyncpg>=0.30.0" \
        "boto3>=1.35.0" \
        "aioboto3>=13.1.0" \
        "tenacity>=9.0.0" \
        "python-ulid>=3.0.0" \
        "orjson>=3.10.0" \
        "python-jose[cryptography]>=3.3.0" \
        "passlib[bcrypt]>=1.7.4"

# ──────────────────────────────────────────── Stage 3: playwright ────────────
FROM builder AS playwright-installer

ENV VIRTUAL_ENV=/build/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Chromium (only) — keeps image size manageable for Fargate
RUN playwright install chromium \
    && playwright install-deps chromium

# ──────────────────────────────────────────── Stage 4: runtime ───────────────
FROM base AS runtime

# Copy virtual environment from builder
COPY --from=builder /build/.venv /app/.venv

# Copy Playwright browsers from playwright-installer stage
# Ensure destination directory exists and has correct ownership
RUN mkdir -p /home/avengine/.cache/ms-playwright \
    && chown -R avengine:avengine /home/avengine/.cache
COPY --from=playwright-installer --chown=avengine:avengine /root/.cache/ms-playwright /home/avengine/.cache/ms-playwright

# Copy application source
COPY --chown=avengine:avengine src/         /app/src/
COPY --chown=avengine:avengine alembic/     /app/alembic/
COPY --chown=avengine:avengine alembic.ini  /app/alembic.ini

# Activate venv — all `python` / `uvicorn` calls use the venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Playwright browser path (non-root)
ENV PLAYWRIGHT_BROWSERS_PATH=/home/avengine/.cache/ms-playwright \
    XDG_CACHE_HOME=/home/avengine/.cache

# Fargate: write scratch/output to ephemeral storage (EFS mount or /tmp)
ENV VIDEO_OUTPUT_DIR=/tmp/avengine/output \
    VIDEO_SCRATCH_DIR=/tmp/avengine/scratch

# Fix permissions on ephemeral dirs
RUN mkdir -p $VIDEO_OUTPUT_DIR $VIDEO_SCRATCH_DIR \
    && chown -R avengine:avengine $VIDEO_OUTPUT_DIR $VIDEO_SCRATCH_DIR

USER 10001

# Health-check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:20000/health')"

EXPOSE 20000

# Use exec form so PID 1 handles SIGTERM correctly (Fargate graceful shutdown)
CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "20000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--no-access-log"]
