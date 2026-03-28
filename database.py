"""
Database stub module.
Provides create_db_and_tables for api.py startup.
Replace with a real database (PostgreSQL/SQLite) for full persistence.
"""

import logging

logger = logging.getLogger(__name__)

# In-memory storage for company tracker
_companies: list = []
_company_updates: list = []
_next_company_id = 1


def create_db_and_tables():
    """Initialize database tables. Currently a no-op stub using in-memory storage."""
    logger.info("Database initialized (in-memory stub)")


def get_companies():
    """Return all tracked companies."""
    return _companies


def add_company(name: str, url: str | None = None) -> dict:
    """Add a new company to track."""
    global _next_company_id
    company = {
        "id": _next_company_id,
        "name": name,
        "url": url,
        "unread_count": 0,
        "last_scanned_at": None,
    }
    _companies.append(company)
    _next_company_id += 1
    return company


def get_company(company_id: int) -> dict | None:
    """Get a company by ID."""
    for c in _companies:
        if c["id"] == company_id:
            return c
    return None


def get_company_updates(company_id: int) -> list:
    """Get updates for a specific company."""
    return [u for u in _company_updates if u["company_id"] == company_id]


def add_company_update(company_id: int, update: dict) -> dict:
    """Add an update for a company."""
    update["company_id"] = company_id
    _company_updates.append(update)
    # Increment unread count
    for c in _companies:
        if c["id"] == company_id:
            c["unread_count"] = c.get("unread_count", 0) + 1
            break
    return update


def mark_updates_read(company_id: int):
    """Mark all updates for a company as read."""
    for u in _company_updates:
        if u["company_id"] == company_id:
            u["is_read"] = True
    for c in _companies:
        if c["id"] == company_id:
            c["unread_count"] = 0
            break
