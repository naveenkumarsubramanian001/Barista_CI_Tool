"""
Scheduler stub module.
Provides start_scheduler for api.py startup.
Replace with APScheduler or similar for real periodic company scanning.
"""

import logging

logger = logging.getLogger(__name__)


def start_scheduler():
    """
    Start the background scheduler for periodic company scanning.
    
    Currently a no-op stub. In production, this would:
    - Periodically scan tracked companies for news updates
    - Use the multi_search_agent to find new articles
    - Store discovered updates in the database
    """
    logger.info(
        "Scheduler initialized (stub mode — periodic scanning disabled). "
        "Implement with APScheduler for production."
    )
