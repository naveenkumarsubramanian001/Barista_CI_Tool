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

    ╔══════════════════════════════════════════════════════════════╗
    ║  ANTI-HALLUCINATION RULES — FOLLOW THESE STRICTLY          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║ 1. ONLY use information explicitly stated in the sources    ║
    ║    provided below. Do NOT invent, fabricate, or assume      ║
    ║    any facts, statistics, dates, quotes, or claims.         ║
    ║ 2. EVERY factual claim MUST cite its source using [N]       ║
    ║    notation where N is the citation_id from the references. ║
    ║ 3. ONLY use citation IDs that exist in the REFERENCES list. ║
    ║    Valid IDs: {valid_citation_ids}                          ║
    ║ 4. If a piece of information is NOT in the provided sources ║
    ║    say "Information not available from current sources"      ║
    ║    instead of guessing or making it up.                     ║
    ║ 5. Do NOT invent company names, product names, revenue      ║
    ║    figures, dates, percentages, or any specific data points ║
    ║    unless they appear verbatim in a source snippet.         ║
    ║ 6. Paraphrase source content accurately. Do NOT exaggerate. ║
    ╚══════════════════════════════════════════════════════════════╝

    === OFFICIAL SOURCES ===
    {official_articles_json}

    === TRUSTED SOURCES ===
    {trusted_articles_json}

    === REFERENCES ===
    {references_json}

    Return ONLY valid JSON (no markdown fences, no commentary):
    {{
      "report_title": "Professional title derived from the query and sources",
      "executive_summary": "3-4 paragraphs synthesizing the sources with [N] inline citations. Stick to facts from sources only.",
      "key_findings": [{{"finding_title": "Theme from sources", "finding_summary": "3-5 sentences summarizing ONLY what sources say, with [N] citations", "source_ids": [1,2]}}],
      "official_insights": [{{"title": "Insight title from official source", "brief_summary": "5-8 sentences paraphrasing ONLY what this source says with [N]", "citation_id": 1}}],
      "trusted_insights": [{{"title": "Insight title from trusted source", "brief_summary": "5-8 sentences paraphrasing ONLY what this source says with [N]", "citation_id": 2}}],
      "cross_source_analysis": "2-3 paragraphs comparing official vs trusted perspectives using ONLY facts from the sources with [N] citations"
    }}

    FINAL RULES:
    - Produce 3-5 key_findings. Each must cite at least 2 real source_ids.
    - EVERY sentence with a factual claim must have at least one [N] citation.
    - cross_source_analysis must compare what officials say vs what trusted analysts say.
    - Do NOT add any text outside the JSON object.
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

    # Use temperature=0.0 for deterministic, non-hallucinated output
    chain = prompt | get_llm(temperature=0.0)

    valid_ids = [str(r["citation_id"]) for r in reference_list]

    with phase_progress("Generating comprehensive report via LLM"):
        response = chain.invoke({
            "query": original_query,
            "official_articles_json": json.dumps(official_data, indent=2),
            "trusted_articles_json": json.dumps(trusted_data, indent=2),
            "references_json": json.dumps(reference_list, indent=2),
            "valid_citation_ids": ", ".join(valid_ids),
        })

    step(1, 2, f"LLM response: {len(response.content)} chars")

    try:
        data = safe_json_extract(response.content)

        report_title = data.get("report_title", f"Research Report: {original_query}")
        executive_summary = data.get("executive_summary", "")
        cross_source_analysis = data.get("cross_source_analysis", "")

        # ─── Post-Generation Hallucination Validation ───────────────
        valid_citation_set = set(range(1, len(all_articles) + 1))

        key_findings_raw = data.get("key_findings", [])
        key_findings = []
        for kf in key_findings_raw:
            if isinstance(kf, dict):
                # Strip invalid citation IDs (hallucinated references)
                raw_ids = kf.get("source_ids", [])
                validated_ids = [sid for sid in raw_ids if isinstance(sid, int) and sid in valid_citation_set]
                if len(validated_ids) < len(raw_ids):
                    warning(f"Stripped {len(raw_ids) - len(validated_ids)} invalid citation IDs from finding '{kf.get('finding_title', '')[:40]}'")
                key_findings.append(KeyFinding(
                    finding_title=kf.get("finding_title", "Finding"),
                    finding_summary=kf.get("finding_summary", ""),
                    source_ids=validated_ids if validated_ids else raw_ids[:2]  # Fallback to first 2
                ))

        official_insights = data.get("official_insights", [])
        trusted_insights = data.get("trusted_insights", [])

        # Validate citation_ids in insights
        for insights_list in [official_insights, trusted_insights]:
            if isinstance(insights_list, list):
                for ins in insights_list:
                    if isinstance(ins, dict) and "citation_id" in ins:
                        cid = ins["citation_id"]
                        if not isinstance(cid, int) or cid not in valid_citation_set:
                            warning(f"Invalid citation_id {cid} in insight '{ins.get('title', '')[:40]}' — clamping to 1")
                            ins["citation_id"] = 1

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
