"""
Companies router for the Company Tracker feature.
Provides CRUD endpoints for tracking competitor companies.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/companies", tags=["companies"])


class CreateCompanyRequest(BaseModel):
    name: str
    url: Optional[str] = None


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
    return company


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


@router.post("/{company_id}/mark-read")
async def mark_read(company_id: int):
    """Mark all updates for a company as read."""
    from database import get_company, mark_updates_read
    company = get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    mark_updates_read(company_id)
    return {"status": "ok"}
