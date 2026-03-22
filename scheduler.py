import asyncio
import secrets
from datetime import datetime, timezone, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlmodel import Session, select
import logging

from database import engine, Company, CompanyUpdate
from graph.workflow import build_graph
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger("company_scraper")

async def scrape_company(company_id: int):
    """Scrapes recent updates for a single company."""
    with Session(engine) as session:
        company = session.get(Company, company_id)
        if not company:
            return
            
        logger.info(f"Scraping updates for {company.name}...")
        days_back = 7 if company.last_scanned_at else 60
        
        if company.url:
            query = f"{company.name} latest news press releases product updates"
        else:
            query = f"{company.name} latest news press releases product updates"
            
        try:
            # Build an isolated graph instance for the background job
            checkpointer = MemorySaver()
            graph = build_graph(checkpointer=checkpointer)
            session_id = f"scrape_{secrets.token_hex(8)}"
            
            initial_state = {
                "original_query": query,
                "company_domains": [company.url] if company.url else [],
                "subqueries": [],
                "official_sources": [],
                "trusted_sources": [],
                "final_ranked_output": {},
                "final_report": None,
                "trusted_domains": [],
                "validation_feedback": "",
                "retry_counts": {"decomposer": 0, "search": 0, "summariser": 0},
                "error": None,
                "search_days_used": None,
                "selected_articles": [],
                "logs": [],
            }
            
            # The pipeline compiles with interrupt_before=["summariser"]
            # To actually force a specific timeframe (days_back), we'd append it to the query, 
            # so the decomposer _extract_days_from_query catches it.
            query_with_days = f"{query} in the last {days_back} days"
            initial_state["original_query"] = query_with_days
            
            # Run the LangGraph workflow up to the interrupt point
            config = {"configurable": {"thread_id": session_id}}
            await graph.ainvoke(initial_state, config=config)
            
            state = graph.get_state(config)
            if state and state.values:
                official = state.values.get("official_sources", [])
                trusted = state.values.get("trusted_sources", [])
                
                all_found = official + trusted
                
                for res in all_found:
                    existing = session.exec(select(CompanyUpdate).where(CompanyUpdate.url == res.get("url"))).first()
                    if not existing:
                        update = CompanyUpdate(
                            company_id=company.id,
                            title=res.get("title", "No Title"),
                            url=res.get("url", ""),
                            snippet=res.get("snippet", "")[:500],
                            source_type=res.get("source_type", "trusted"),
                            published_date=res.get("published_date", datetime.now(timezone.utc).isoformat()),
                            is_read=False
                        )
                        session.add(update)
            
        except Exception as e:
            logger.error(f"Error scraping for {company.name}: {e}")

        company.last_scanned_at = datetime.now(timezone.utc)
        session.add(company)
        session.commit()

async def scrape_companies_job():
    """Background job to scrape recent updates for all companies."""
    logger.info("Starting company scraper job...")
    with Session(engine) as session:
        companies = session.exec(select(Company)).all()
        for company in companies:
            await scrape_company(company.id)
    logger.info("Finished company scraper job.")

scheduler = AsyncIOScheduler()

def start_scheduler():
    scheduler.add_job(scrape_companies_job, 'interval', days=7, id="scrape_companies_job", replace_existing=True)
    scheduler.start()
    logger.info("Started APScheduler for background scraping.")
