"""
Celery application factory.

Workers run the pipeline tasks in isolated processes, bypassing the Python
GIL for CPU-bound FFmpeg work.  Beat scheduler handles RSS monitoring.
"""
from __future__ import annotations

from celery import Celery
from celery.signals import worker_process_init
from kombu import Exchange, Queue

from src.core.config import get_settings

settings = get_settings()

# ── App ───────────────────────────────────────────────────────────────────────
celery_app = Celery(
    "avengine",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "src.pipeline.tasks",      # pipeline dispatch task (Step 3)
        "src.agents.sourcing.tasks",  # crawler beat tasks (Step 4)
    ],
)

# ── Configuration ─────────────────────────────────────────────────────────────
celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Timeouts
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,         # one task per worker at a time (heavy video jobs)
    # Result expiry
    result_expires=86400,                 # 24 h
    # Routing
    task_default_queue="pipeline",
    task_queues=[
        Queue("pipeline", Exchange("pipeline"), routing_key="pipeline"),
        Queue("render",   Exchange("render"),   routing_key="render"),
        Queue("crawl",    Exchange("crawl"),     routing_key="crawl"),
    ],
    task_routes={
        "src.pipeline.tasks.run_pipeline":        {"queue": "pipeline"},
        "src.agents.rendering.tasks.render_video": {"queue": "render"},
        "src.agents.sourcing.tasks.crawl_feeds":  {"queue": "crawl"},
    },
    # Beat schedule (passive RSS monitoring — Step 4)
    beat_schedule={
        "crawl-rss-feeds": {
            "task": "src.agents.sourcing.tasks.crawl_feeds",
            "schedule": 300.0,  # every 5 minutes
        },
    },
)


@worker_process_init.connect
def _init_worker(**kwargs: object) -> None:  # noqa: ANN003
    """
    Bootstrap logging + Sentry inside each worker process.

    Celery forks *after* the main process, so signal handlers must
    re-initialise resources that don't survive a fork.
    """
    import logging
    import os
    import pathlib

    from src.core.logging import configure_logging

    configure_logging(
        log_level=settings.log_level.value,
        json_logs=settings.is_production,
    )

    # Celery prefork children are separate OS processes; systemd only captures
    # the main-process stdout.  Write child-process logs to a file so they are
    # visible (tail -f logs/celery_worker.log).
    log_dir = pathlib.Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "celery_worker.log")
    if logging.root.handlers:
        file_handler.setFormatter(logging.root.handlers[0].formatter)
    else:
        file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.root.addHandler(file_handler)

    # Inject API keys into os.environ so LangChain/OpenAI SDK clients pick them up
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key

    if settings.sentry_dsn:
        import sentry_sdk

        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.environment.value,
        )
