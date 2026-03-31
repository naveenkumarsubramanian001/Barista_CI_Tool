import inspect
import json
import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse

from agents.discriminators import search_discriminator, summariser_discriminator
from agents.multi_search_agent import multi_search_agent
from agents.summariser import summariser_agent
from database import (
    add_company_update,
    add_notification,
    add_report_event,
    get_company,
    get_company_updates_by_ids,
    update_company_scan_telemetry,
    update_company_scan_timestamps,
)
from models.schemas import Article, ResearchState
from nodes.rank_filter import rank_filter_node
from utils.geturl import find_official_domains
from utils.pdf_report import generate_pdf


logger = logging.getLogger(__name__)


COMPANY_QUERY_TEMPLATES = [
    "{company} latest news",
    "{company} product launch",
    "{company} press release",
    "{company} partnership announcement",
    "{company} funding",
    "{company} acquisition",
    "{company} technology update",
    "{company} AI development",
    "{company} leadership changes",
    "{company} strategic partnerships",
]

DEFAULT_TRUSTED_DOMAINS = [
    "reuters.com",
    "techcrunch.com",
    "theverge.com",
    "wired.com",
    "venturebeat.com",
    "thenextweb.com",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _extract_domain_from_url(url: str | None) -> str | None:
    if not url:
        return None
    candidate = url.strip()
    if not candidate:
        return None
    if "://" not in candidate:
        candidate = f"https://{candidate}"

    try:
        host = (urlparse(candidate).hostname or "").strip().lower()
    except Exception:
        return None

    if not host:
        return None

    if host.startswith("www."):
        host = host[4:]
    return host or None


def _article_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _to_article(item: dict) -> Article | None:
    url = (item.get("url") or "").strip()
    if not url:
        return None

    try:
        return Article(
            title=item.get("title") or "Untitled",
            url=url,
            snippet=item.get("snippet") or "",
            published_date=item.get("published_date") or _utc_now().strftime("%Y-%m-%d"),
            source_type=item.get("source_type") or "trusted",
            domain=item.get("domain") or None,
            priority=bool(item.get("priority", False)),
            score=float(item.get("score") or 0.0),
        )
    except Exception:
        return None


async def _run_node(fn, state: ResearchState) -> ResearchState:
    result = fn(state)
    if inspect.isawaitable(result):
        result = await result
    return result


def build_company_queries(company_name: str) -> list[str]:
    name = company_name.strip()
    return [template.format(company=name) for template in COMPANY_QUERY_TEMPLATES]


async def run_company_tracking_scan(
    company_id: int,
    *,
    search_days: int,
    trigger: str,
    create_notifications: bool,
) -> dict:
    started = time.perf_counter()

    company = get_company(company_id)
    if not company:
        duration_ms = int((time.perf_counter() - started) * 1000)
        update_company_scan_telemetry(
            company_id,
            last_run_status="failed",
            last_error="Company not found",
            last_duration_ms=duration_ms,
            last_trigger=trigger,
        )
        raise ValueError("Company not found")

    company_name = (company.get("name") or "").strip()
    if not company_name:
        duration_ms = int((time.perf_counter() - started) * 1000)
        update_company_scan_telemetry(
            company_id,
            last_run_status="failed",
            last_error="Company name is empty",
            last_duration_ms=duration_ms,
            last_trigger=trigger,
        )
        raise ValueError("Company name is empty")

    official_domain = _extract_domain_from_url(company.get("url"))
    if not official_domain:
        domains = await find_official_domains([company_name])
        official_domain = domains[0] if domains else None

    state: ResearchState = {
        "original_query": f"{company_name} company intelligence monitoring",
        "subqueries": build_company_queries(company_name),
        "official_sources": [],
        "trusted_sources": [],
        "final_ranked_output": {},
        "final_report": None,
        "company_domains": [official_domain] if official_domain else [],
        "trusted_domains": DEFAULT_TRUSTED_DOMAINS,
        "primary_entity": company_name,
        "guardrail_status": "valid",
        "guardrail_reason": "Company tracking scan — pre-approved",
        "guardrail_blocked": False,
        "validation_feedback": "",
        "validation_passed": False,
        "validation_metrics": {},
        "decomposition_score": 0.0,
        "redundancy_pairs": [],
        "coverage_gaps": [],
        "semantic_warnings": [],
        "retry_counts": {"decomposer": 0, "search": 0, "summariser": 0, "report_guardrail": 0},
        "error": None,
        "search_days_used": int(search_days),
        "selected_articles": [],
        "logs": [f"Company tracking scan started ({trigger})."],
        "stages": [],
        "current_stage": "collect",
        "progress_percentage": 0,
    }

    try:
        state = await _run_node(multi_search_agent, state)
        state = await _run_node(search_discriminator, state)
        state = await _run_node(rank_filter_node, state)

        final_ranked = state.get("final_ranked_output") or {}
        official = final_ranked.get("official_sources") or []
        trusted = final_ranked.get("trusted_sources") or []
        ranked_articles = [*official, *trusted]

        inserted = 0
        for item in ranked_articles:
            update = {
                "title": _article_value(item, "title", "Untitled"),
                "url": _article_value(item, "url"),
                "snippet": _article_value(item, "snippet"),
                "source_type": _article_value(item, "source_type", "trusted"),
                "published_date": _article_value(item, "published_date"),
                "is_read": False,
                "metadata": {
                    "domain": _article_value(item, "domain"),
                    "score": _article_value(item, "score"),
                    "priority": _article_value(item, "priority", False),
                    "trigger": trigger,
                },
            }
            created = add_company_update(company_id, update)
            if created:
                inserted += 1

        now = _utc_now()
        update_company_scan_timestamps(
            company_id,
            last_scanned_at=now.isoformat(),
            next_scanned_at=(now + timedelta(days=7)).isoformat(),
        )

        duration_ms = int((time.perf_counter() - started) * 1000)
        update_company_scan_telemetry(
            company_id,
            last_run_status="success",
            last_error=None,
            last_duration_ms=duration_ms,
            last_trigger=trigger,
        )

        if create_notifications and inserted > 0:
            add_notification(
                title=f"New intelligence found for {company_name}",
                message=f"{inserted} new insights available for review",
                company_id=company_id,
            )

        return {
            "company_id": company_id,
            "company_name": company_name,
            "trigger": trigger,
            "search_days": search_days,
            "total_ranked": len(ranked_articles),
            "new_insights": inserted,
            "duration_ms": duration_ms,
        }
    except Exception as exc:
        duration_ms = int((time.perf_counter() - started) * 1000)
        update_company_scan_telemetry(
            company_id,
            last_run_status="failed",
            last_error=str(exc),
            last_duration_ms=duration_ms,
            last_trigger=trigger,
        )
        logger.exception("Company scan failed for company_id=%s", company_id)
        if create_notifications:
            add_notification(
                title=f"Scan failed for {company_name}",
                message=f"{type(exc).__name__}: {exc}",
                company_id=company_id,
            )
        raise


async def generate_company_report(company_id: int, update_ids: list[int]) -> dict:
    started = time.perf_counter()
    company = get_company(company_id)
    if not company:
        raise ValueError("Company not found")

    updates = get_company_updates_by_ids(company_id, update_ids)
    selected_articles: list[Article] = []

    for update in updates:
        article = _to_article(update)
        if article:
            selected_articles.append(article)

    if not selected_articles:
        raise ValueError("No valid selected insights with URLs were found")

    official = [a for a in selected_articles if a.source_type == "official"]
    trusted = [a for a in selected_articles if a.source_type != "official"]

    state: ResearchState = {
        "original_query": f"{company['name']} intelligence report",
        "subqueries": [],
        "official_sources": official,
        "trusted_sources": trusted,
        "final_ranked_output": {
            "official_sources": official,
            "trusted_sources": trusted,
        },
        "final_report": None,
        "company_domains": [],
        "trusted_domains": [],
        "primary_entity": company["name"],
        "guardrail_status": "valid",
        "guardrail_reason": "Company report generation — pre-approved",
        "guardrail_blocked": False,
        "validation_feedback": "",
        "validation_passed": False,
        "validation_metrics": {},
        "decomposition_score": 0.0,
        "redundancy_pairs": [],
        "coverage_gaps": [],
        "semantic_warnings": [],
        "retry_counts": {"decomposer": 0, "search": 0, "summariser": 0, "report_guardrail": 0},
        "error": None,
        "search_days_used": 30,
        "selected_articles": [a.url for a in selected_articles],
        "logs": ["Generating company report from selected insights."],
        "stages": [],
        "current_stage": "prepare",
        "progress_percentage": 96,
    }

    state = await _run_node(summariser_agent, state)
    state = await _run_node(summariser_discriminator, state)

    final_report = state.get("final_report")
    if not final_report:
        raise RuntimeError("Failed to generate final report")

    session_id = f"company_{company_id}_{secrets.token_hex(6)}"
    report_json_name = f"report_{session_id}.json"
    report_pdf_name = f"report_{session_id}.pdf"

    with open(report_json_name, "w", encoding="utf-8") as file_handle:
        json.dump(final_report, file_handle, indent=2)
    generate_pdf(report_json_name, report_pdf_name)

    add_report_event(
        company_id,
        session_id=session_id,
        report_json=report_json_name,
        report_pdf=report_pdf_name,
        selected_update_ids=update_ids,
    )

    add_notification(
        title=f"Report generated for {company['name']}",
        message=f"Selected insights were compiled into report {session_id}",
        company_id=company_id,
    )

    duration_ms = int((time.perf_counter() - started) * 1000)

    return {
        "company_id": company_id,
        "session_id": session_id,
        "report": final_report,
        "pdf_url": f"/api/pdf/download/{session_id}",
        "duration_ms": duration_ms,
    }
