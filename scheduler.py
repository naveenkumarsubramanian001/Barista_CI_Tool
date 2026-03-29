"""Background scheduler for weekly tracked-company monitoring."""

import asyncio
import logging
import threading
from typing import Awaitable, Callable

from database import get_due_companies

logger = logging.getLogger(__name__)

_RUNNER_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()
_SCAN_HANDLER: Callable[..., Awaitable[dict]] | None = None


def register_scan_handler(handler: Callable[..., Awaitable[dict]]) -> None:
    """Register async callback used for scheduled company scans."""
    global _SCAN_HANDLER
    _SCAN_HANDLER = handler


async def _run_single_scheduled_scan(company_id: int, semaphore: asyncio.Semaphore, task_timeout_seconds: int):
    if _SCAN_HANDLER is None:
        return

    async with semaphore:
        try:
            logger.info("Running scheduled scan for company_id=%s", company_id)
            await asyncio.wait_for(
                _SCAN_HANDLER(
                    company_id=company_id,
                    search_days=7,
                    trigger="scheduled",
                    create_notifications=True,
                ),
                timeout=task_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.error("Scheduled scan timed out for company_id=%s", company_id)
        except Exception as exc:
            logger.exception("Scheduled scan failed for company_id=%s: %s", company_id, exc)


async def _scheduler_loop(
    poll_seconds: int,
    max_concurrency: int,
    task_timeout_seconds: int,
):
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    while not _STOP_EVENT.is_set():
        if _SCAN_HANDLER is None:
            await asyncio.sleep(poll_seconds)
            continue

        try:
            due_companies = get_due_companies()
            tasks = [
                asyncio.create_task(
                    _run_single_scheduled_scan(
                        int(company["id"]),
                        semaphore,
                        task_timeout_seconds,
                    )
                )
                for company in due_companies
            ]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as exc:
            logger.exception("Scheduler loop error: %s", exc)

        await asyncio.sleep(poll_seconds)


def _run_scheduler_thread(
    poll_seconds: int,
    max_concurrency: int,
    task_timeout_seconds: int,
):
    asyncio.run(_scheduler_loop(poll_seconds, max_concurrency, task_timeout_seconds))


def start_scheduler(
    poll_seconds: int = 60,
    max_concurrency: int = 3,
    task_timeout_seconds: int = 180,
):
    """Start background scheduler thread once per process."""
    global _RUNNER_THREAD
    if _RUNNER_THREAD and _RUNNER_THREAD.is_alive():
        logger.info("Scheduler already running")
        return

    _STOP_EVENT.clear()
    _RUNNER_THREAD = threading.Thread(
        target=_run_scheduler_thread,
        args=(poll_seconds, max_concurrency, task_timeout_seconds),
        daemon=True,
        name="company-tracking-scheduler",
    )
    _RUNNER_THREAD.start()
    logger.info("Scheduler initialized (weekly company monitoring enabled)")
