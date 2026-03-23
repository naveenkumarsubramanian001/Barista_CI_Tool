"""
Comprehensive Report Summariser with Rich logging.
"""

import json
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ResearchState, FinalReport, Insight, KeyFinding
from config import get_llm
from utils.logger import (
    section, info, success, warning, error, detail,
    step, report_summary, phase_progress
)


def summariser_agent(state: ResearchState) -> ResearchState:
    """Summarizes top articles into a comprehensive, cited research report."""
    section("Report Generation", "📋")

    final_ranked_output = state.get("final_ranked_output", {})
    official_top = final_ranked_output.get("official_sources", [])
    trusted_top = final_ranked_output.get("trusted_sources", [])
    original_query = state.get("original_query", "")
    all_articles = state.get("official_sources", []) + state.get("trusted_sources", [])

    if not official_top and not trusted_top:
        warning("No ranked articles to summarise")
        return state

    info(f"Building report from {len(official_top)} official + {len(trusted_top)} trusted sources")

    reference_list = []
    for i, article in enumerate(all_articles):
        reference_list.append({
            "citation_id": i + 1, "title": article.title, "url": article.url,
            "domain": article.domain, "published_date": article.published_date,
            "source_type": article.source_type
        })

    prompt = ChatPromptTemplate.from_template("""
    You are an expert competitive intelligence analyst creating a comprehensive research report.
    Create a COMPLETE, PROFESSIONAL research report on "{query}" that reads like a yearly industry report.
    Use [citation_id] notation (e.g., [1], [2]) to cite sources inline.

    === OFFICIAL SOURCES ===
    {official_articles_json}

    === TRUSTED SOURCES ===
    {trusted_articles_json}

    === REFERENCES ===
    {references_json}

    Return valid JSON:
    {{
      "report_title": "Professional title",
      "executive_summary": "3-4 paragraphs synthesizing ALL sources with [N] citations",
      "key_findings": [{{"finding_title": "Theme", "finding_summary": "3-5 sentences with [N] citations", "source_ids": [1,2]}}],
      "official_insights": [{{"title": "Insight", "brief_summary": "5-8 sentences with [N]", "citation_id": 1}}],
      "trusted_insights": [{{"title": "Insight", "brief_summary": "5-8 sentences with [N]", "citation_id": 2}}],
      "cross_source_analysis": "2-3 paragraphs comparing official vs trusted with [N] citations"
    }}

    RULES: 3-5 key_findings, EVERY claim must cite [N], cross_source_analysis must be analytical.
    """)

    from utils.json_utils import safe_json_extract

    official_data = []
    trusted_data = []

    for article in official_top:
        try:
            citation_id = all_articles.index(article) + 1
        except ValueError:
            citation_id = 1
        official_data.append({
            "citation_id": citation_id, "title": article.title,
            "snippet": article.snippet, "domain": article.domain,
            "published_date": article.published_date, "url": article.url
        })

    for article in trusted_top:
        try:
            citation_id = all_articles.index(article) + 1
        except ValueError:
            citation_id = 1
        trusted_data.append({
            "citation_id": citation_id, "title": article.title,
            "snippet": article.snippet, "domain": article.domain,
            "published_date": article.published_date, "url": article.url
        })

    chain = prompt | get_llm()

    with phase_progress("Generating comprehensive report via LLM"):
        response = chain.invoke({
            "query": original_query,
            "official_articles_json": json.dumps(official_data, indent=2),
            "trusted_articles_json": json.dumps(trusted_data, indent=2),
            "references_json": json.dumps(reference_list, indent=2)
        })

    step(1, 2, f"LLM response: {len(response.content)} chars")

    try:
        data = safe_json_extract(response.content)

        report_title = data.get("report_title", f"Research Report: {original_query}")
        executive_summary = data.get("executive_summary", "")
        cross_source_analysis = data.get("cross_source_analysis", "")

        key_findings_raw = data.get("key_findings", [])
        key_findings = []
        for kf in key_findings_raw:
            if isinstance(kf, dict):
                key_findings.append(KeyFinding(
                    finding_title=kf.get("finding_title", "Finding"),
                    finding_summary=kf.get("finding_summary", ""),
                    source_ids=kf.get("source_ids", [])
                ))

        official_insights = data.get("official_insights", [])
        trusted_insights = data.get("trusted_insights", [])

        if not official_insights and official_top:
            for k, v in data.items():
                if "official" in k.lower() and isinstance(v, list):
                    official_insights = v
                    break

        if not trusted_insights and trusted_top:
            for k, v in data.items():
                if "trusted" in k.lower() and isinstance(v, list):
                    trusted_insights = v
                    break
            if not trusted_insights:
                for k, v in data.items():
                    if "insight" in k.lower() and isinstance(v, list) and k != "official_insights":
                        trusted_insights = v
                        break

        valid_data = {
            "report_title": report_title,
            "executive_summary": executive_summary,
            "key_findings": [kf.model_dump() for kf in key_findings],
            "official_insights": official_insights,
            "trusted_insights": trusted_insights,
            "cross_source_analysis": cross_source_analysis,
            "references": [a.model_dump() for a in all_articles]
        }

        report = FinalReport(**valid_data)
        state["final_report"] = report.model_dump()

        step(2, 2, "Report assembled")
        report_summary(report.model_dump())

    except Exception as e:
        state["error"] = f"Summariser failed: {str(e)}"
        error(f"Summariser failed: {e}")

    return state
