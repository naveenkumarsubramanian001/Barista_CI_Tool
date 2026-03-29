"""
Companies router for the Company Tracker feature.
Provides CRUD endpoints for tracking competitor companies.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/companies", tags=["companies"])
logger = logging.getLogger(__name__)


class CreateCompanyRequest(BaseModel):
    name: str
    url: Optional[str] = None


class GenerateCompanyReportRequest(BaseModel):
    update_ids: list[int]


async def _run_initial_company_scan(company_id: int, company_name: str):
    from database import add_notification
    from services.company_tracking import run_company_tracking_scan

    try:
        await run_company_tracking_scan(
            company_id=company_id,
            search_days=30,
            trigger="initial",
            create_notifications=True,
        )
    except Exception as exc:
        logger.exception("Initial company scan failed for company_id=%s", company_id)
        add_notification(
            title=f"Initial scan failed for {company_name}",
            message=f"{type(exc).__name__}: {exc}",
            company_id=company_id,
        )


@router.get("/")
async def list_companies():
    """List all tracked companies."""
    from database import get_companies
    return get_companies()


@router.post("/")
async def create_company(request: CreateCompanyRequest):
    """Add a new company to track."""
    from database import add_company
    company = add_company(name=request.name, url=request.url)

    # Initial intelligence collection (last 30 days) starts immediately.
    asyncio.create_task(_run_initial_company_scan(int(company["id"]), company["name"]))

    return company


@router.get("/notifications")
async def list_notifications(limit: int = 50, unread_only: bool = False):
    from database import get_notifications, get_unread_notification_count

    notifications = get_notifications(limit=limit, unread_only=unread_only)
    return {
        "notifications": notifications,
        "unread_count": get_unread_notification_count(),
    }


@router.post("/notifications/{notification_id}/read")
async def mark_notification_as_read(notification_id: int):
    from database import mark_notification_read

    if not mark_notification_read(notification_id):
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "ok"}


@router.get("/{company_id}")
async def get_company_detail(company_id: int):
    """Get details for a specific tracked company."""
    from database import get_company
    company = get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company


@router.get("/{company_id}/updates")
async def get_updates(company_id: int):
    """Get news/updates for a specific company."""
    from database import get_company, get_company_updates
    company = get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    updates = get_company_updates(company_id)
    return {"company": company, "updates": updates}


@router.get("/{company_id}/reports")
async def get_reports(company_id: int):
    """Get generated report events for a specific company."""
    from database import get_company, get_company_report_events

    company = get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    reports = get_company_report_events(company_id)
    return {"company": company, "reports": reports}


@router.post("/{company_id}/mark-read")
async def mark_read(company_id: int):
    """Mark all updates for a company as read."""
    from database import get_company, mark_updates_read
    company = get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    mark_updates_read(company_id)
    return {"status": "ok"}


@router.post("/{company_id}/updates/{update_id}/read")
async def mark_single_update_read(company_id: int, update_id: int):
    from database import get_company, mark_update_read

    company = get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    updated = mark_update_read(company_id, update_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Update not found")

    return {"status": "ok"}


@router.post("/{company_id}/scrape")
async def trigger_company_search(company_id: int):
    from database import get_company
    from services.company_tracking import run_company_tracking_scan

    company = get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    result = await run_company_tracking_scan(
        company_id=company_id,
        search_days=30,
        trigger="manual",
        create_notifications=True,
    )
    return {"status": "ok", **result}


@router.post("/{company_id}/generate-report")
async def generate_report_from_updates(company_id: int, request: GenerateCompanyReportRequest):
    from database import get_company
    from services.company_tracking import generate_company_report

    company = get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    if not request.update_ids:
        raise HTTPException(status_code=400, detail="Select at least one insight")

    try:
        result = await generate_company_report(company_id, request.update_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {exc}")

    return {"status": "ok", **result}


