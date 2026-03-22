import asyncio
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session, select

from database import Company, CompanyUpdate, get_session
from pydantic import BaseModel
import secrets

router = APIRouter(prefix="/api/companies", tags=["companies"])

class CompanyCreate(BaseModel):
    name: str
    url: Optional[str] = None

class CompanyResponse(BaseModel):
    id: int
    name: str
    url: Optional[str]
    unread_count: int
    last_scanned_at: Optional[str]
    next_scanned_at: Optional[str]

@router.post("/", response_model=CompanyResponse)
async def create_company(company_in: CompanyCreate, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    # Check if company exists
    company = session.exec(select(Company).where(Company.name == company_in.name)).first()
    if company:
        raise HTTPException(status_code=400, detail="Company already registered")
    
    new_company = Company(name=company_in.name, url=company_in.url)
    session.add(new_company)
    session.commit()
    session.refresh(new_company)
    
    # Trigger the background 60 days scraping job here
    from scheduler import scrape_company
    background_tasks.add_task(scrape_company, new_company.id)
    
    return CompanyResponse(
        id=new_company.id,
        name=new_company.name,
        url=new_company.url,
        unread_count=0,
        last_scanned_at=None,
        next_scanned_at=None
    )

@router.get("/", response_model=List[CompanyResponse])
def get_companies(session: Session = Depends(get_session)):
    companies = session.exec(select(Company)).all()
    results = []
    for c in companies:
        unread = session.exec(select(CompanyUpdate).where(CompanyUpdate.company_id == c.id, CompanyUpdate.is_read == False)).all()
        
        last_scan = c.last_scanned_at.isoformat() if c.last_scanned_at else None
        
        from datetime import timedelta
        next_scan = (c.last_scanned_at + timedelta(days=7)).isoformat() if c.last_scanned_at else None
        
        results.append(CompanyResponse(
            id=c.id, 
            name=c.name, 
            url=c.url, 
            unread_count=len(unread),
            last_scanned_at=last_scan,
            next_scanned_at=next_scan
        ))
    return results

@router.get("/{company_id}/updates", response_model=List[CompanyUpdate])
def get_company_updates(company_id: int, session: Session = Depends(get_session)):
    updates = session.exec(select(CompanyUpdate).where(CompanyUpdate.company_id == company_id).order_by(CompanyUpdate.created_at.desc())).all()
    return updates

@router.post("/{company_id}/scrape")
def trigger_manual_scrape(company_id: int, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    company = session.get(Company, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    from scheduler import scrape_company
    background_tasks.add_task(scrape_company, company.id)
    return {"status": "started", "company_id": company.id}

@router.post("/{company_id}/updates/{update_id}/read")
def mark_update_read(company_id: int, update_id: int, session: Session = Depends(get_session)):
    update = session.get(CompanyUpdate, update_id)
    if not update or update.company_id != company_id:
        raise HTTPException(status_code=404, detail="Update not found")
    
    update.is_read = True
    session.add(update)
    session.commit()
    return {"status": "success"}

class GenerateReportRequest(BaseModel):
    update_ids: List[int]

@router.post("/{company_id}/generate-report")
async def generate_report_for_company(company_id: int, request: GenerateReportRequest, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    company = session.get(Company, company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    if not request.update_ids:
        raise HTTPException(status_code=400, detail="No updates selected")
        
    updates = session.exec(select(CompanyUpdate).where(CompanyUpdate.id.in_(request.update_ids))).all()
    if not updates:
        raise HTTPException(status_code=400, detail="Invalid updates selected")

    session_id = f"sess_{secrets.token_hex(8)}"
    config = {"configurable": {"thread_id": session_id}}
    
    official_articles = []
    trusted_articles = []
    for u in updates:
        domain = u.url.split("/")[2] if "//" in u.url else u.url
        art_dict = {
            "url": str(u.url), 
            "title": str(u.title), 
            "snippet": str(u.snippet), 
            "domain": domain,
            "published_date": str(u.published_date) if hasattr(u, "published_date") and u.published_date else "Recent",
            "source_type": str(u.source_type) if hasattr(u, "source_type") and u.source_type else "trusted"
        }
        
        if art_dict["source_type"] == "official":
            official_articles.append(art_dict)
        else:
            trusted_articles.append(art_dict)
        
    state = {
        "original_query": f"Company Intelligence Report: {company.name}",
        "final_ranked_output": {
            "official_sources": official_articles,
            "trusted_sources": trusted_articles
        },
        "official_sources": official_articles,
        "trusted_sources": trusted_articles,
        "selected_articles": [u.url for u in updates]
    }
    
    from api import graph_app
    # Set the state as if the 'ranker' node just finished.
    # Because of interrupt_before=["summariser"], this will leave the graph paused at summariser.
    graph_app.update_state(config, state, as_node="ranker")
    
    async def run_report_generation():
        print(f"[{session_id}] 🚀 Starting Phase 2 (Summarise & PDF) for Company Tracker...")
        try:
            final_output = await graph_app.ainvoke(None, config=config)
            if final_output and final_output.get("final_report"):
                import json
                report_json_name = f"report_{session_id}.json"
                report_pdf_name = f"report_{session_id}.pdf"
                
                with open(report_json_name, "w") as f:
                    json.dump(final_output["final_report"], f, indent=2)
                
                from utils.pdf_report import generate_pdf as gen_pdf_file
                gen_pdf_file(report_json_name, report_pdf_name)
                print(f"[{session_id}] ✅ Report generated successfully.")
            else:
                print(f"[{session_id}] ❌ Failed to generate report data.")
        except Exception as e:
            print(f"[{session_id}] ❌ Error generating report: {e}")

    background_tasks.add_task(run_report_generation)

    return {"status": "started", "session_id": session_id, "company_id": company.id}
