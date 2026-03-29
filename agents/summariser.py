"""Summariser with overview + per-article insight format (Barista CI style)."""

import json
import re
from datetime import datetime
from typing import Any

import requests
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate

from config import get_llm
from models.schemas import Article, FinalReport, KeyFinding, ResearchState
from utils.json_utils import safe_json_extract
from utils.logger import error, info, phase_progress, report_summary, section, step, warning


SAFE_REQUEST_TOKEN_LIMIT = 9000
CHARS_PER_TOKEN_ESTIMATE = 4
MAX_ARTICLE_CHARS = 3600
MIN_ARTICLE_CHARS = 1200
PER_ARTICLE_TOKEN_BUDGET = 2800
OVERVIEW_TOKEN_BUDGET = 4200
MAX_SELECTED_ARTICLES = 8
_RATE_LIMIT_BLOCK_UNTIL: datetime | None = None


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def _trim_text_to_token_budget(text: str, token_budget: int) -> str:
    if not text:
        return ""
    max_chars = max(0, token_budget * CHARS_PER_TOKEN_ESTIMATE)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].strip()


def _compact_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + "..."


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "ratelimiterror" in msg
        or "rate limit" in msg
        or "rate_limit_exceeded" in msg
        or "tokens per day" in msg
        or "error code: 429" in msg
    )


def _parse_wait_seconds(message: str) -> int:
    # Example: "Please try again in 42m44.352s"
    m = re.search(r"in\s+(?:(\d+)m)?([\d.]+)s", message)
    if not m:
        return 45 * 60
    minutes = int(m.group(1) or 0)
    seconds = float(m.group(2) or 0.0)
    total = int(minutes * 60 + seconds)
    return max(total, 60)


def _register_rate_limit(exc: Exception) -> None:
    global _RATE_LIMIT_BLOCK_UNTIL
    wait_seconds = _parse_wait_seconds(str(exc))
    _RATE_LIMIT_BLOCK_UNTIL = datetime.utcnow().fromtimestamp(
        datetime.utcnow().timestamp() + wait_seconds
    )


def _llm_blocked() -> bool:
    if _RATE_LIMIT_BLOCK_UNTIL is None:
        return False
    return datetime.utcnow() < _RATE_LIMIT_BLOCK_UNTIL


def _to_articles(items: list[Any]) -> list[Article]:
    result: list[Article] = []
    for item in items:
        if isinstance(item, Article):
            result.append(item)
        elif isinstance(item, dict):
            try:
                result.append(Article(**item))
            except Exception:
                continue
    return result


def _extract_article_text(url: str, timeout: int = 10) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        res = requests.get(url, headers=headers, timeout=timeout)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()

        article_tag = soup.find("article") or soup.find("main")
        if article_tag:
            text_content = article_tag.get_text(separator="\n", strip=True)
        else:
            paragraphs = soup.find_all("p")
            text_content = "\n\n".join(
                p.get_text(strip=True)
                for p in paragraphs
                if len(p.get_text(strip=True)) > 60
            )

        if not text_content:
            return ""
        return text_content[:MAX_ARTICLE_CHARS]
    except Exception:
        return ""


def _normalize_insight(raw: dict[str, Any], citation_id: int, source_url: str) -> dict[str, Any]:
    detailed_summary = _compact_text(str(raw.get("detailed_summary", "")).strip(), 1400)
    reasoning = _compact_text(str(raw.get("reasoning", "")).strip(), 700)
    sentiment = _compact_text(str(raw.get("sentiment", "Neutral")).strip(), 80)

    key_metrics = raw.get("key_metrics", [])
    if not isinstance(key_metrics, list):
        key_metrics = []
    key_features = raw.get("key_features", [])
    if not isinstance(key_features, list):
        key_features = []

    brief_summary = _compact_text(
        str(raw.get("brief_summary", detailed_summary)).strip(),
        650,
    )
    if not brief_summary:
        brief_summary = f"Summary unavailable for citation [{citation_id}]"

    key_findings = []
    key_findings.extend([_compact_text(str(item), 180) for item in key_metrics[:4]])
    key_findings.extend([_compact_text(str(item), 180) for item in key_features[:4]])
    if not key_findings:
        key_findings.append(f"No structured findings extracted for citation [{citation_id}]")

    return {
        "title": _compact_text(str(raw.get("title", "Untitled Insight")), 180),
        "brief_summary": brief_summary,
        "detailed_summary": detailed_summary,
        "reasoning": reasoning,
        "sentiment": sentiment,
        "key_metrics": [_compact_text(str(item), 120) for item in key_metrics[:6]],
        "key_features": [_compact_text(str(item), 120) for item in key_features[:6]],
        "citation_id": citation_id,
        "tags": [f"#Sentiment:{sentiment.replace(' ', '')}"],
        "overview": _compact_text(detailed_summary or brief_summary, 700),
        "key_findings": key_findings[:6],
        "strategic_analysis": reasoning,
        "analysis": reasoning,
        "why_it_matters": reasoning,
        "business_impact": reasoning,
        "practical_significance": reasoning,
        "technical_context": "; ".join([_compact_text(str(item), 100) for item in key_features[:3]]),
        "source_url": source_url,
    }


def _summarize_single_article(query: str, payload: dict[str, Any]) -> dict[str, Any]:
    citation_id = payload["citation_id"]
    title = payload.get("title", "Untitled")
    url = payload.get("url", "")
    domain = payload.get("domain", "unknown")
    snippet = payload.get("snippet", "")
    content = payload.get("extracted_text") or snippet or "No content available"

    content = _trim_text_to_token_budget(content, PER_ARTICLE_TOKEN_BUDGET)
    if len(content) < MIN_ARTICLE_CHARS and snippet:
        content = f"{snippet}\n\n{content}".strip()[:MAX_ARTICLE_CHARS]

    estimated = _estimate_tokens(f"{query}\n{title}\n{domain}\n{content}") + 700
    if estimated > SAFE_REQUEST_TOKEN_LIMIT:
        content = _trim_text_to_token_budget(content, max(900, SAFE_REQUEST_TOKEN_LIMIT - 700))

    insight_prompt = ChatPromptTemplate.from_template(
        """
You are an elite competitive intelligence analyst. You are evaluating a single article about "{query}".

Article Domain: {domain}
Article Title: {title}

Full Text Content:
{content}

Your job is to extract critical facts, strategy, numbers, and features.
Do not hallucinate. Do not write about other articles.

Return only valid JSON:
{{
  "title": "string",
  "detailed_summary": "string",
  "reasoning": "string",
  "sentiment": "string",
  "key_metrics": ["string"],
  "key_features": ["string"]
}}
"""
    )

    chain = insight_prompt | get_llm(temperature=0.0)
    if _llm_blocked():
        fallback = {
            "title": title,
            "detailed_summary": f"{_compact_text(snippet or content, 900)} [{citation_id}]",
            "reasoning": f"Generated with non-LLM fallback due to temporary LLM rate limit. [{citation_id}]",
            "sentiment": "Neutral",
            "key_metrics": [],
            "key_features": [],
        }
        return _normalize_insight(fallback, citation_id, source_url=url)

    try:
        response = chain.invoke(
            {
                "query": query,
                "domain": domain,
                "title": title,
                "content": content,
            }
        )
        parsed = safe_json_extract(response.content)
        return _normalize_insight(parsed, citation_id, source_url=url)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            _register_rate_limit(exc)
            warning("LLM rate limit reached. Falling back to snippet-based insight summarization.")
        warning(f"Per-article summarization failed for citation [{citation_id}]: {exc}")
        fallback = {
            "title": title,
            "detailed_summary": f"{_compact_text(snippet or content, 900)} [{citation_id}]",
            "reasoning": f"This source contributes evidence for the query. [{citation_id}]",
            "sentiment": "Neutral",
            "key_metrics": [],
            "key_features": [],
        }
        return _normalize_insight(fallback, citation_id, source_url=url)


def _build_overview_fallback(
    query: str,
    company_name: str,
    generated_on: str,
    generated_time: str,
    official_snippets: list[dict[str, Any]],
    trusted_snippets: list[dict[str, Any]],
) -> dict[str, Any]:
    official_titles = [s.get("title", "") for s in official_snippets[:3] if s.get("title")]
    trusted_titles = [s.get("title", "") for s in trusted_snippets[:3] if s.get("title")]

    executive_summary = (
        f"This report summarizes selected developments for '{query}'. "
        f"A total of {len(official_snippets)} official and {len(trusted_snippets)} trusted insights were analyzed. "
        f"Current synthesis was generated with fallback mode because the LLM daily token limit was reached; "
        f"content remains grounded in selected sources."
    )

    conflict_and_consensus = (
        "Official sources emphasize first-party announcements and strategic intent, while trusted sources "
        "add external interpretation and market context. Consensus exists on key development themes, with "
        "differences mainly in framing and impact emphasis."
    )

    key_findings = [
        {
            "finding_title": "Selected-source synthesis",
            "finding_summary": "Report assembled from human-selected official and trusted sources.",
            "source_ids": [1, 2],
        }
    ]

    return {
        "report_title": f"Competitive Intelligence Report: {query}",
        "company_name": company_name,
        "query_topic": query,
        "generated_on": generated_on,
        "generated_time": generated_time,
        "executive_summary": executive_summary,
        "conflict_and_consensus": conflict_and_consensus,
        "introduction": (
            f"This briefing analyzes selected developments related to {company_name}. "
            f"Official highlights include: {', '.join(official_titles) if official_titles else 'selected first-party announcements'}. "
            f"Trusted coverage includes: {', '.join(trusted_titles) if trusted_titles else 'selected third-party assessments'}."
        ),
        "strategic_significance": "The selected developments indicate active strategic motion in product, partnerships, and positioning.",
        "research_scope": (
            f"Insights analyzed: {len(official_snippets) + len(trusted_snippets)}. "
            f"Official: {len(official_snippets)}. Trusted: {len(trusted_snippets)}."
        ),
        "official_intelligence": "Official sources provide direct statements of priorities, launches, and strategic commitments.",
        "market_context": "Trusted sources provide external framing, comparative context, and implications for competition.",
        "report_structure": "The report presents executive summary, categorized insights, synthesis, and references.",
        "cross_source_analysis": conflict_and_consensus,
        "conclusion": "Overall direction appears consistent with expansion and competitive positioning through selected strategic actions.",
        "analysis_summary": "The aggregate signal indicates a coordinated strategy supported by multiple source types.",
        "official_strategic_signals": "First-party communications emphasize planned capabilities and strategic initiatives.",
        "independent_market_assessment": "Third-party narratives generally support the direction while adding risk/opportunity interpretation.",
        "temporal_significance": "Recency of selected items indicates active, ongoing execution rather than historical-only positioning.",
        "key_takeaways": [
            "Selected official and trusted insights are directionally aligned.",
            "External framing complements first-party strategic messaging.",
        ],
        "recommended_actions": [
            "Track follow-on announcements and execution milestones.",
            "Benchmark competitor responses over the next review cycle.",
        ],
        "key_findings": key_findings,
    }


def _build_overview(
    query: str,
    company_name: str,
    generated_on: str,
    generated_time: str,
    official_snippets: list[dict[str, Any]],
    trusted_snippets: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    if _llm_blocked():
        return _build_overview_fallback(
            query=query,
            company_name=company_name,
            generated_on=generated_on,
            generated_time=generated_time,
            official_snippets=official_snippets,
            trusted_snippets=trusted_snippets,
        )

    prompt = ChatPromptTemplate.from_template(
        """
Create a professional competitive intelligence report overview on "{query}".

Rules:
- Use only provided sources.
- Do not hallucinate.
- Return only valid JSON.

Query: {query}
Company: {company_name}
Generated on: {generated_on}
Generated time: {generated_time}

Official Sources (Titles & Snippets):
{official_snippets}

Trusted Sources (Titles & Snippets):
{trusted_snippets}

References:
{references_json}

Return JSON:
{{
  "report_title": "Report Title / Topic",
  "company_name": "...",
  "query_topic": "...",
  "generated_on": "...",
  "generated_time": "...",
  "executive_summary": "150-250 words",
  "conflict_and_consensus": "Differences and agreements between official and trusted sources",
  "introduction": "250-350 words briefing paragraph",
  "strategic_significance": "...",
  "research_scope": "...",
  "official_intelligence": "...",
  "market_context": "...",
  "report_structure": "...",
  "cross_source_analysis": "150-200 words",
  "conclusion": "200-300 words synthesis",
  "analysis_summary": "...",
  "official_strategic_signals": "...",
  "independent_market_assessment": "...",
  "temporal_significance": "...",
  "key_takeaways": ["...", "..."],
  "recommended_actions": ["...", "..."],
  "key_findings": [
    {{"finding_title": "...", "finding_summary": "...", "source_ids": [1,2]}}
  ]
}}
"""
    )

    payload = {
        "query": query,
        "company_name": company_name,
        "generated_on": generated_on,
        "generated_time": generated_time,
        "official_snippets": json.dumps(official_snippets, separators=(",", ":")),
        "trusted_snippets": json.dumps(trusted_snippets, separators=(",", ":")),
        "references_json": json.dumps(references, separators=(",", ":")),
    }

    estimated = _estimate_tokens(
        payload["query"]
        + payload["official_snippets"]
        + payload["trusted_snippets"]
        + payload["references_json"]
    ) + 800
    if estimated > OVERVIEW_TOKEN_BUDGET:
        official_small = [
            {"title": item.get("title", ""), "snippet": _compact_text(item.get("snippet", ""), 160)}
            for item in official_snippets
        ]
        trusted_small = [
            {"title": item.get("title", ""), "snippet": _compact_text(item.get("snippet", ""), 160)}
            for item in trusted_snippets
        ]
        payload["official_snippets"] = json.dumps(official_small, separators=(",", ":"))
        payload["trusted_snippets"] = json.dumps(trusted_small, separators=(",", ":"))

    chain = prompt | get_llm(temperature=0.0)
    try:
        response = chain.invoke(payload)
        return safe_json_extract(response.content)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            _register_rate_limit(exc)
            warning("LLM rate limit reached during overview generation. Using deterministic fallback overview.")
            return _build_overview_fallback(
                query=query,
                company_name=company_name,
                generated_on=generated_on,
                generated_time=generated_time,
                official_snippets=official_snippets,
                trusted_snippets=trusted_snippets,
            )
        raise


def summariser_agent(state: ResearchState) -> ResearchState:
    """Summarize selected insights and build final report."""
    section("Report Generation", "📋")

    final_ranked_output = state.get("final_ranked_output", {})
    official_passed = _to_articles(final_ranked_output.get("official_sources", []))
    trusted_passed = _to_articles(final_ranked_output.get("trusted_sources", []))
    selected_urls = set(state.get("selected_articles", []))
    original_query = state.get("original_query", "")

    now_utc = datetime.utcnow()
    generated_on = now_utc.strftime("%Y-%m-%d")
    generated_time = now_utc.strftime("%H:%M UTC")

    if selected_urls:
        official_selected = [a for a in official_passed if a.url in selected_urls]
        trusted_selected = [a for a in trusted_passed if a.url in selected_urls]
    else:
        official_selected = [a for a in official_passed if getattr(a, "priority", False)] or official_passed[:3]
        trusted_selected = [a for a in trusted_passed if getattr(a, "priority", False)] or trusted_passed[:3]

    selected_articles = official_selected + trusted_selected
    if len(selected_articles) > MAX_SELECTED_ARTICLES:
        selected_articles = selected_articles[:MAX_SELECTED_ARTICLES]
        selected_url_set = {a.url for a in selected_articles}
        official_selected = [a for a in official_selected if a.url in selected_url_set]
        trusted_selected = [a for a in trusted_selected if a.url in selected_url_set]

    if not selected_articles:
        warning("No selected insights available for final report generation")
        state["error"] = "No selected insights available for report generation."
        return state

    info(
        f"Building report from selected insights: {len(official_selected)} official + "
        f"{len(trusted_selected)} trusted"
    )

    reference_list = []
    for i, article in enumerate(selected_articles):
        reference_list.append(
            {
                "citation_id": i + 1,
                "title": article.title,
                "url": article.url,
                "domain": article.domain,
                "published_date": article.published_date,
                "source_type": article.source_type,
                "snippet": article.snippet,
            }
        )

    citation_map = {article.url: i + 1 for i, article in enumerate(selected_articles)}
    company_name = "Target Company"
    if selected_articles:
        domain = str(getattr(selected_articles[0], "domain", "")).strip()
        if domain:
            company_name = domain.split(".")[0].replace("-", " ").title()

    unique_domains = sorted(
        {
            str(ref.get("domain", "")).strip()
            for ref in reference_list
            if str(ref.get("domain", "")).strip()
        }
    )
    domain_text = ", ".join(unique_domains[:10]) if unique_domains else "N/A"
    report_header = "\n".join(
        [
            f"{original_query or 'Competitive Intelligence Topic'}",
            f"Generated: {generated_on} | {generated_time}",
            f"Total Sources Analysed: {len(reference_list)}",
            f"Official Insights: {len(official_selected)} | Trusted Insights: {len(trusted_selected)}",
            "Classification: INTERNAL USE -- R&D",
            "Produced by Barista Competitive Intelligence Tool",
        ]
    )

    step(1, 4, "Extracting article bodies")
    official_payload = []
    trusted_payload = []

    for article in official_selected:
        official_payload.append(
            {
                "citation_id": citation_map.get(article.url, 1),
                "title": article.title,
                "url": article.url,
                "domain": article.domain,
                "snippet": article.snippet,
                "extracted_text": _extract_article_text(article.url),
            }
        )

    for article in trusted_selected:
        trusted_payload.append(
            {
                "citation_id": citation_map.get(article.url, 1),
                "title": article.title,
                "url": article.url,
                "domain": article.domain,
                "snippet": article.snippet,
                "extracted_text": _extract_article_text(article.url),
            }
        )

    step(2, 4, "Building per-article insights")
    official_insights = [_summarize_single_article(original_query, p) for p in official_payload]
    trusted_insights = [_summarize_single_article(original_query, p) for p in trusted_payload]

    step(3, 4, "Building executive overview")
    with phase_progress("Generating overview via LLM"):
        overview_data = _build_overview(
            query=original_query,
            company_name=company_name,
            generated_on=generated_on,
            generated_time=generated_time,
            official_snippets=[{"title": a["title"], "snippet": a["brief_summary"]} for a in official_insights],
            trusted_snippets=[{"title": a["title"], "snippet": a["brief_summary"]} for a in trusted_insights],
            references=reference_list,
        )

    try:
        valid_citation_set = set(range(1, len(selected_articles) + 1))

        key_findings_raw = overview_data.get("key_findings", [])
        key_findings: list[KeyFinding] = []
        for item in key_findings_raw:
            if not isinstance(item, dict):
                continue
            source_ids = [
                sid
                for sid in item.get("source_ids", [])
                if isinstance(sid, int) and sid in valid_citation_set
            ]
            key_findings.append(
                KeyFinding(
                    finding_title=item.get("finding_title", "Finding"),
                    finding_summary=item.get("finding_summary", ""),
                    source_ids=source_ids,
                )
            )

        if not key_findings:
            key_findings = [
                KeyFinding(
                    finding_title="Selected Insights Summary",
                    finding_summary="Insights were derived from selected official and trusted sources.",
                    source_ids=list(valid_citation_set)[:3],
                )
            ]

        valid_data = {
            "report_title": overview_data.get(
                "report_title",
                f"Competitive Intelligence Report: {original_query}",
            ),
            "company_name": overview_data.get("company_name", company_name),
            "query_topic": overview_data.get("query_topic", original_query),
            "generated_on": overview_data.get("generated_on", generated_on),
            "generated_time": overview_data.get("generated_time", generated_time),
            "report_header": report_header,
            "introduction": overview_data.get("introduction", ""),
            "strategic_significance": overview_data.get("strategic_significance", ""),
            "research_scope": overview_data.get(
                "research_scope",
                f"Insights: {len(selected_articles)} from {len(reference_list)} sources across domains: {domain_text}.",
            ),
            "official_intelligence": overview_data.get("official_intelligence", ""),
            "market_context": overview_data.get("market_context", ""),
            "report_structure": overview_data.get("report_structure", ""),
            "executive_summary": overview_data.get("executive_summary", ""),
            "conflict_and_consensus": overview_data.get(
                "conflict_and_consensus",
                overview_data.get("cross_source_analysis", ""),
            ),
            "key_findings": [k.model_dump() for k in key_findings],
            "official_insights": official_insights,
            "trusted_insights": trusted_insights,
            "cross_source_analysis": overview_data.get("cross_source_analysis", ""),
            "conclusion": overview_data.get("conclusion", ""),
            "analysis_summary": overview_data.get("analysis_summary", ""),
            "official_strategic_signals": overview_data.get("official_strategic_signals", ""),
            "independent_market_assessment": overview_data.get("independent_market_assessment", ""),
            "temporal_significance": overview_data.get("temporal_significance", ""),
            "key_takeaways": overview_data.get("key_takeaways", []),
            "recommended_actions": overview_data.get("recommended_actions", []),
            "references": [a.model_dump() for a in selected_articles],
        }

        report = FinalReport(**valid_data)
        state["final_report"] = report.model_dump()
        state["logs"].append("Report generated from human-selected insights.")

        step(4, 4, "Report assembled")
        report_summary(report.model_dump())
    except Exception as exc:
        state["error"] = f"Summariser failed: {exc}"
        error(f"Summariser failed: {exc}")

    return state
