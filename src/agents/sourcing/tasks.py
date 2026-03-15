"""
Celery beat tasks for the Sourcing agent.

  crawl_feeds  — scheduled every 5 minutes (see core/celery_app.py beat_schedule)
                 Runs the async RSSMonitor and logs discovery stats.
"""
from __future__ import annotations

import asyncio

import structlog

from src.core.celery_app import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(
    name="src.agents.sourcing.tasks.crawl_feeds",
    queue="crawl",
    max_retries=1,
    default_retry_delay=60,
    ignore_result=True,
)
def crawl_feeds() -> None:
    """
    Periodic RSS/feed crawl.

    Runs the full async RSSMonitor cycle, which:
      - Polls all registered feeds
      - Scrapes article HTML for embedded video URLs
      - Deduplicates against Redis seen-set
      - Pushes new URLs to the `sourcing:discovered` Redis queue
    """
    from src.agents.sourcing.rss_monitor import RSSMonitor

    log = logger.bind(task="crawl_feeds")
    log.info("crawl_feeds.start")

    try:
        monitor = RSSMonitor()
        discovered = asyncio.run(monitor.run())
        log.info("crawl_feeds.complete", discovered=len(discovered))
    except Exception as exc:  # noqa: BLE001
        log.error("crawl_feeds.failed", error=str(exc))
        raise
