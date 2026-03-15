AVEngine — Installation & Usage Guide
Prerequisites
Requirement	Minimum Version	Notes
Python	3.11+	pyenv recommended
Docker + Docker Compose	24+	For local stack
FFmpeg	6.0+	Must be on $PATH
Node.js	—	Not required
uv	latest	Fast Python package manager
Install uv and FFmpeg:


curl -LsSf https://astral.sh/uv/install.sh | sh
# Ubuntu / Debian
sudo apt-get install ffmpeg
# macOS
brew install ffmpeg
1. Clone & Install

git clone <your-repo-url> AVEngine
cd AVEngine

# Create virtual environment and install all dependencies
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Install Playwright's Chromium browser
playwright install chromium
playwright install-deps chromium
2. Environment Configuration

cp .env.example .env
Open .env and fill in the required values. The minimum required to boot:


# .env — absolute minimum to run locally
SECRET_KEY=replace-with-32-plus-chars-random-string!!

# LLM (at least one required for scripting)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# TwelveLabs (required for matching)
TWELVELABS_API_KEY=tlk_...
TWELVELABS_INDEX_ID=                 # leave blank — auto-created on first run

# ElevenLabs (optional — pipeline degrades gracefully without it)
ELEVENLABS_API_KEY=...

# AWS S3 (optional — videos saved locally if not set)
AWS_S3_BUCKET=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# Proxies (optional — yt-dlp works without them, just slower)
PROXY_ENDPOINT=
PROXY_USERNAME=
PROXY_PASSWORD=
3. Start the Local Infrastructure
This starts Postgres, Redis, and Qdrant via Docker Compose:


docker compose up postgres redis qdrant -d

# Verify all three are healthy
docker compose ps
Expected output:


NAME        STATUS
postgres    running (healthy)
redis       running (healthy)
qdrant      running
4. Database Migrations

# Initialise Alembic (first time only)
alembic init alembic

# Generate the initial migration from the ORM models
alembic revision --autogenerate -m "initial"

# Apply migrations
alembic upgrade head
Tip: If you get Can't find module 'env', ensure your alembic.ini has script_location = alembic and the env.py imports from src.core.database import Base and from src.core.models import Task.

5. Run the Application
You need three separate terminals:

Terminal 1 — API server:


uvicorn src.main:app --reload --host 0.0.0.0 --port 20000
Terminal 2 — Pipeline + render worker:


celery -A src.core.celery_app worker \
  --loglevel=info \
  --queues=pipeline,render \
  --concurrency=2
Terminal 3 — Crawler beat worker (optional, for RSS monitoring):


celery -A src.core.celery_app worker \
  --loglevel=info \
  --queues=crawl \
  --concurrency=1 \
  --beat
Or use Docker Compose to run everything at once:


# Build image first
docker compose build

# Start the full stack
docker compose up
6. Verify the Installation

# Liveness probe
curl http://localhost:8000/health

# Readiness probe (checks Postgres, Redis, Qdrant)
curl http://localhost:8000/health/ready
Expected response:


{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development",
  "components": {
    "postgres": {"status": "healthy", "latency_ms": 1.2},
    "redis":    {"status": "healthy", "latency_ms": 0.4},
    "qdrant":   {"status": "healthy", "latency_ms": 2.1}
  }
}
Interactive API docs (dev mode only):


http://localhost:8000/docs
7. Generate Your First Video
Step A — Submit a job:


curl -s -X POST http://localhost:8000/api/v1/generate-video \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "How large language models are changing software engineering",
    "style": "explainer",
    "duration_seconds": 60,
    "aspect_ratio": "16:9",
    "target_audience": "software engineers"
  }' | jq .
Response (202 Accepted):


{
  "task_id": "01JK2MXYZ...",
  "status": "queued",
  "status_url": "http://localhost:8000/api/v1/tasks/01JK2MXYZ...",
  "ws_url":     "ws://localhost:8000/api/v1/ws/tasks/01JK2MXYZ..."
}
Step B — Poll for status:


TASK_ID="01JK2MXYZ..."

watch -n 3 "curl -s http://localhost:8000/api/v1/tasks/$TASK_ID | jq '{status,stage,progress}'"
Step C — Stream real-time events (WebSocket):


# Using websocat (brew install websocat / apt install websocat)
websocat ws://localhost:8000/api/v1/ws/tasks/$TASK_ID
Each event looks like:


{"task_id":"01JK2MXYZ...","type":"scripting","message":"Crew is researching and writing the script...","progress":20.0}
{"task_id":"01JK2MXYZ...","type":"sourcing","message":"Sourcing B-roll footage...","progress":35.0}
{"task_id":"01JK2MXYZ...","type":"rendering","message":"Rendering final video...","progress":95.0}
{"task_id":"01JK2MXYZ...","type":"done","message":"Video ready.","progress":100.0,"data":{"video_url":"..."}}
Step D — Retrieve the result:


curl -s http://localhost:8000/api/v1/tasks/$TASK_ID/result | jq .

{
  "task_id": "01JK2MXYZ...",
  "status": "completed",
  "video_url": "https://your-bucket.s3.us-east-1.amazonaws.com/videos/.../output.mp4",
  "thumbnail_url": "https://...",
  "duration_seconds": 63.4
}
Cancel a running job:


curl -X DELETE http://localhost:8000/api/v1/tasks/$TASK_ID
8. Configuration Reference
Variable	Default	Purpose
OPENAI_MODEL	gpt-4o	Model for the Researcher agent
ANTHROPIC_MODEL	claude-sonnet-4-6	Writer + Editor + Pydantic AI extractor
TWELVELABS_INDEX_ID	(auto-created)	Leave blank on first run
ELEVENLABS_VOICE_ID	21m00Tcm4TlvDq8ikWAM	Rachel voice — change to any ElevenLabs voice ID
FFMPEG_THREADS	4	CPU threads for FFmpeg encode
VIDEO_MAX_DURATION_SECONDS	300	Hard cap on output video length
PROXY_ENDPOINT	—	host:port for residential proxy provider
QDRANT_COLLECTION_NAME	video_embeddings	Qdrant collection for clip vectors
9. Running Tests

# All tests with coverage
pytest

# Unit tests only (no external services needed)
pytest tests/unit -v

# Integration tests (requires running Postgres + Redis)
pytest tests/integration -v

# Single module
pytest tests/unit/test_extractor.py -v
10. Production Deployment (AWS Fargate)

# Build and push the production image
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

docker build --target runtime -t avengine:latest .
docker tag avengine:latest <account>.dkr.ecr.us-east-1.amazonaws.com/avengine:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/avengine:latest
Recommended Fargate task sizing:

Service	CPU	Memory	Notes
API	0.5 vCPU	1 GB	Stateless, scale horizontally
Pipeline worker	2 vCPU	4 GB	1 task per Celery worker
Render worker	4 vCPU	8 GB	FFmpeg is CPU-intensive
Crawler worker	0.5 vCPU	1 GB	Low-intensity beat tasks
Required AWS services:

RDS (Postgres 16) — DATABASE_URL
ElastiCache (Redis 7) — REDIS_URL, CELERY_BROKER_URL
S3 — output video + thumbnail storage
Secrets Manager — inject all *_API_KEY values as environment variables
EFS — mount at /tmp/avengine for scratch files shared between API + worker tasks