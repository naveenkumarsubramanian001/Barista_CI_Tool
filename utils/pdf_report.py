import json
import os
import textwrap
from typing import Any

import fitz


PAGE_WIDTH = 595
PAGE_HEIGHT = 842
MARGIN_LEFT = 42
MARGIN_RIGHT = 42
MARGIN_TOP = 46
MARGIN_BOTTOM = 44
CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT

TEXT_COLOR = (0.15, 0.18, 0.23)
MUTED_TEXT_COLOR = (0.37, 0.42, 0.48)
PRIMARY_COLOR = (0.09, 0.29, 0.52)
SECTION_BG_COLOR = (0.93, 0.96, 1.0)
LINK_COLOR = (0.04, 0.36, 0.79)
DIVIDER_COLOR = (0.78, 0.82, 0.88)

FONT_REG = "Helvetica"
FONT_BOLD = "Helvetica-Bold"


def _safe(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _new_page(doc: fitz.Document) -> tuple[fitz.Page, float]:
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    return page, MARGIN_TOP


def _line_height(font_size: float) -> float:
    return font_size * 1.45


def _wrap_paragraph(
    text: str,
    font_size: float,
    bullet: str | None = None,
    available_width: float = CONTENT_WIDTH,
) -> list[str]:
    clean = " ".join(_safe(text).split())
    if not clean:
        return []

    approx_char_width = max(1.0, font_size * 0.52)
    max_chars = max(24, int(available_width / approx_char_width))

    if bullet:
        body_lines = textwrap.wrap(clean, width=max_chars - len(bullet) - 1)
        if not body_lines:
            return [bullet]
        out = [f"{bullet} {body_lines[0]}"]
        indent = " " * (len(bullet) + 1)
        out.extend([f"{indent}{line}" for line in body_lines[1:]])
        return out

    return textwrap.wrap(clean, width=max_chars)


def _ensure_space(doc: fitz.Document, page: fitz.Page, y: float, needed: float) -> tuple[fitz.Page, float]:
    if y + needed <= PAGE_HEIGHT - MARGIN_BOTTOM:
        return page, y
    return _new_page(doc)


def _draw_divider(page: fitz.Page, y: float) -> float:
    page.draw_line(
        p1=(MARGIN_LEFT, y),
        p2=(PAGE_WIDTH - MARGIN_RIGHT, y),
        color=DIVIDER_COLOR,
        width=0.8,
    )
    return y + 10


def _write_link(
    doc: fitz.Document,
    page: fitz.Page,
    y: float,
    url: str,
    label: str | None = None,
    font_size: float = 10.2,
    x: float = MARGIN_LEFT,
) -> tuple[fitz.Page, float]:
    url_clean = _safe(url)
    if not url_clean:
        return page, y

    text = _safe(label) or url_clean
    max_width = PAGE_WIDTH - MARGIN_RIGHT - x
    lines = _wrap_paragraph(text, font_size, available_width=max_width)
    if not lines:
        return page, y

    lh = _line_height(font_size)
    needed = len(lines) * lh + 2
    page, y = _ensure_space(doc, page, y, needed)

    for line in lines:
        page.insert_text((x, y), line, fontsize=font_size, fontname=FONT_REG, color=LINK_COLOR)
        line_width = fitz.get_text_length(line, fontname=FONT_REG, fontsize=font_size)
        rect = fitz.Rect(x, y - font_size, x + line_width, y + 2)
        try:
            page.insert_link({"kind": fitz.LINK_URI, "from": rect, "uri": url_clean})
        except Exception:
            # Keep rendering even if a malformed URL cannot be attached as a link.
            pass
        y += lh

    return page, y + 2


def _write_footer(page: fitz.Page, page_number: int, page_count: int) -> None:
    footer = f"Barista CI Report   |   Page {page_number}/{page_count}"
    y = PAGE_HEIGHT - 20
    page.draw_line(
        p1=(MARGIN_LEFT, y - 10),
        p2=(PAGE_WIDTH - MARGIN_RIGHT, y - 10),
        color=DIVIDER_COLOR,
        width=0.6,
    )
    page.insert_text(
        (MARGIN_LEFT, y),
        footer,
        fontsize=8.6,
        fontname=FONT_REG,
        color=MUTED_TEXT_COLOR,
    )


def _write_heading(
    doc: fitz.Document,
    page: fitz.Page,
    y: float,
    text: str,
    font_size: float = 15,
    divider: bool = True,
) -> tuple[fitz.Page, float]:
    lines = _wrap_paragraph(text, font_size, available_width=CONTENT_WIDTH - 16)
    needed = max(1, len(lines)) * _line_height(font_size) + (16 if divider else 8)
    page, y = _ensure_space(doc, page, y, needed)

    band_top = y - font_size
    band_bottom = y + max(1, len(lines)) * _line_height(font_size) + 2
    page.draw_rect(
        fitz.Rect(MARGIN_LEFT - 4, band_top, PAGE_WIDTH - MARGIN_RIGHT + 4, band_bottom),
        color=SECTION_BG_COLOR,
        fill=SECTION_BG_COLOR,
        width=0,
    )

    for line in lines:
        page.insert_text((MARGIN_LEFT + 4, y), line, fontsize=font_size, fontname=FONT_BOLD, color=PRIMARY_COLOR)
        y += _line_height(font_size)

    if divider:
        y = _draw_divider(page, y - 2)
    else:
        y += 4
    return page, y


def _write_subheading(
    doc: fitz.Document,
    page: fitz.Page,
    y: float,
    text: str,
    font_size: float = 11.5,
) -> tuple[fitz.Page, float]:
    lines = _wrap_paragraph(text, font_size, available_width=CONTENT_WIDTH - 18)
    needed = max(1, len(lines)) * _line_height(font_size) + 4
    page, y = _ensure_space(doc, page, y, needed)

    for line in lines:
        page.insert_text((MARGIN_LEFT + 10, y), line, fontsize=font_size, fontname=FONT_BOLD, color=PRIMARY_COLOR)
        y += _line_height(font_size)
    y += 2
    return page, y


def _write_paragraph(
    doc: fitz.Document,
    page: fitz.Page,
    y: float,
    text: str,
    font_size: float = 10.5,
    x: float = MARGIN_LEFT + 12,
    color: tuple[float, float, float] = TEXT_COLOR,
) -> tuple[fitz.Page, float]:
    lines = _wrap_paragraph(text, font_size, available_width=PAGE_WIDTH - MARGIN_RIGHT - x)
    if not lines:
        return page, y + 8

    needed = len(lines) * _line_height(font_size) + 4
    page, y = _ensure_space(doc, page, y, needed)
    for line in lines:
        page.insert_text((x, y), line, fontsize=font_size, fontname=FONT_REG, color=color)
        y += _line_height(font_size)
    y += 2
    return page, y


def _write_bullets(
    doc: fitz.Document,
    page: fitz.Page,
    y: float,
    items: list[str],
    font_size: float = 10.2,
) -> tuple[fitz.Page, float]:
    if not items:
        return _write_paragraph(doc, page, y, "No data available.", font_size=font_size)

    for item in items:
        lines = _wrap_paragraph(item, font_size, bullet="•", available_width=CONTENT_WIDTH - 16)
        needed = len(lines) * _line_height(font_size) + 2
        page, y = _ensure_space(doc, page, y, needed)
        for line in lines:
            page.insert_text((MARGIN_LEFT + 14, y), line, fontsize=font_size, fontname=FONT_REG, color=TEXT_COLOR)
            y += _line_height(font_size)
    y += 2
    return page, y


def _render_insights_section(
    doc: fitz.Document,
    page: fitz.Page,
    y: float,
    heading: str,
    description: str,
    insights: list[dict],
    references: list[dict],
) -> tuple[fitz.Page, float]:
    page, y = _write_heading(doc, page, y, heading)
    page, y = _write_paragraph(doc, page, y, description, color=MUTED_TEXT_COLOR)

    for idx, insight in enumerate(insights, start=1):
        citation_id = insight.get("citation_id", idx)
        title = _safe(insight.get("title"), "Untitled Insight")

        page, y = _write_subheading(doc, page, y, f"{title}  -  [Citation {citation_id}]")

        if isinstance(citation_id, int) and 1 <= citation_id <= len(references):
            ref = references[citation_id - 1]
            citation_url = _safe(insight.get("source_url")) or _safe(ref.get("url"))
            if citation_url:
                page, y = _write_link(doc, page, y, citation_url, label=f"Open citation [{citation_id}]")

        tags = insight.get("tags") or []
        if isinstance(tags, list) and tags:
            page, y = _write_paragraph(
                doc,
                page,
                y,
                " ".join(str(tag) for tag in tags if str(tag).strip()),
                font_size=9.8,
                color=MUTED_TEXT_COLOR,
            )

        page, y = _write_subheading(doc, page, y, "Overview", font_size=10.8)
        page, y = _write_paragraph(
            doc,
            page,
            y,
            _safe(insight.get("overview"), _safe(insight.get("brief_summary"), "Information not available.")),
        )

        page, y = _write_subheading(doc, page, y, "Key Findings", font_size=10.8)
        key_findings = insight.get("key_findings") or []
        if isinstance(key_findings, list) and key_findings:
            page, y = _write_bullets(doc, page, y, [_safe(item) for item in key_findings])
        else:
            page, y = _write_bullets(doc, page, y, ["Information not available from current sources."])

        page, y = _write_subheading(doc, page, y, "Analysis", font_size=10.8)
        page, y = _write_paragraph(
            doc,
            page,
            y,
            _safe(
                insight.get("strategic_analysis") or insight.get("analysis"),
                "Information not available from current sources.",
            ),
        )

        page, y = _write_subheading(doc, page, y, "Why It Matters", font_size=10.8)
        page, y = _write_paragraph(
            doc,
            page,
            y,
            _safe(insight.get("why_it_matters"), "Information not available from current sources."),
        )

        page, y = _write_subheading(doc, page, y, "Practical Significance", font_size=10.8)
        page, y = _write_paragraph(
            doc,
            page,
            y,
            _safe(
                insight.get("business_impact") or insight.get("practical_significance"),
                "Information not available from current sources.",
            ),
        )

        page, y = _write_subheading(doc, page, y, "Technical Context", font_size=10.8)
        page, y = _write_paragraph(
            doc,
            page,
            y,
            _safe(insight.get("technical_context"), "Information not available from current sources."),
        )

        if isinstance(citation_id, int) and 1 <= citation_id <= len(references):
            ref = references[citation_id - 1]
            ref_url = _safe(insight.get("source_url")) or _safe(ref.get("url"))
            if ref_url:
                page, y = _write_subheading(doc, page, y, "Source", font_size=10.8)
                page, y = _write_link(doc, page, y, ref_url)

        y = _draw_divider(page, y)

    return page, y


def generate_pdf(json_path: str, pdf_path: str) -> str:
    """Generate a readable PDF from a report JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to read report JSON: {exc}") from exc

    os.makedirs(os.path.dirname(pdf_path) or ".", exist_ok=True)

    title = _safe(report.get("report_title"), "Competitive Intelligence Report")
    query_topic = _safe(report.get("query_topic"), title)
    generated_on = _safe(report.get("generated_on"), "")
    generated_time = _safe(report.get("generated_time"), "")
    report_header = _safe(report.get("report_header"), "")
    introduction = _safe(report.get("introduction"), "")
    strategic_significance = _safe(report.get("strategic_significance"), "")
    research_scope = _safe(report.get("research_scope"), "")
    official_intelligence = _safe(report.get("official_intelligence"), "")
    market_context = _safe(report.get("market_context"), "")
    report_structure = _safe(report.get("report_structure"), "")
    executive_summary = _safe(report.get("executive_summary"), "No executive summary available.")
    official_insights = report.get("official_insights") or []
    trusted_insights = report.get("trusted_insights") or []
    analysis_summary = _safe(report.get("analysis_summary"), "")
    official_signals = _safe(report.get("official_strategic_signals"), "")
    market_assessment = _safe(report.get("independent_market_assessment"), "")
    temporal_significance = _safe(report.get("temporal_significance"), "")
    key_takeaways = report.get("key_takeaways") or []
    recommended_actions = report.get("recommended_actions") or []
    conclusion = _safe(report.get("conclusion"), "")
    references = report.get("references") or []

    doc = fitz.open()

    page, y = _new_page(doc)

    # Title block
    page, y = _write_heading(doc, page, y, title, font_size=18)
    header_text = report_header or "\n".join(
        [
            f"{query_topic}",
            f"Generated: {generated_on} | {generated_time}",
            f"Total Sources Analysed: {len(references)}",
            f"Official Insights: {len(official_insights)} | Trusted Insights: {len(trusted_insights)}",
            "Classification: INTERNAL USE -- R&D",
            "Produced by Barista Competitive Intelligence Tool",
        ]
    )
    for row in header_text.split("\n"):
        page, y = _write_paragraph(doc, page, y, row, font_size=10.2, color=MUTED_TEXT_COLOR)
    y = _draw_divider(page, y)

    page, y = _write_heading(doc, page, y, "1. Executive Summary", font_size=14)
    page, y = _write_paragraph(doc, page, y, executive_summary)

    page, y = _write_heading(doc, page, y, "2. Introduction", font_size=14)
    page, y = _write_paragraph(doc, page, y, introduction)
    page, y = _write_subheading(doc, page, y, "Strategic Significance")
    page, y = _write_paragraph(doc, page, y, strategic_significance)
    page, y = _write_subheading(doc, page, y, "Research Scope")
    page, y = _write_paragraph(doc, page, y, research_scope)
    page, y = _write_subheading(doc, page, y, "Official Intelligence")
    page, y = _write_paragraph(doc, page, y, official_intelligence)
    page, y = _write_subheading(doc, page, y, "Market Context")
    page, y = _write_paragraph(doc, page, y, market_context)
    page, y = _write_subheading(doc, page, y, "Report Structure")
    page, y = _write_paragraph(doc, page, y, report_structure)

    if official_insights:
        page, y = _render_insights_section(
            doc,
            page,
            y,
            f"3. Insights from Official Sources ({len(official_insights)} articles)",
            "Official sources are first-party channels such as company blogs, press releases, and official announcements.",
            official_insights,
            references,
        )

    if trusted_insights:
        page, y = _render_insights_section(
            doc,
            page,
            y,
            f"4. Insights from Trusted Sources ({len(trusted_insights)} articles)",
            "Trusted sources include independent journalism, technology publications, analysts, and industry observers.",
            trusted_insights,
            references,
        )

    page, y = _write_heading(doc, page, y, "5. Conclusion & Recommendations", font_size=14)
    page, y = _write_subheading(doc, page, y, "Analysis Summary")
    page, y = _write_paragraph(doc, page, y, analysis_summary)
    page, y = _write_subheading(doc, page, y, "Official Strategic Signals")
    page, y = _write_paragraph(doc, page, y, official_signals)
    page, y = _write_subheading(doc, page, y, "Independent Market Assessment")
    page, y = _write_paragraph(doc, page, y, market_assessment)
    page, y = _write_subheading(doc, page, y, "Temporal Significance")
    page, y = _write_paragraph(doc, page, y, temporal_significance)
    page, y = _write_subheading(doc, page, y, "Key Takeaways & Implications")
    takeaways = [
        _safe(item)
        for item in key_takeaways
        if _safe(item)
    ] if isinstance(key_takeaways, list) else []
    page, y = _write_bullets(doc, page, y, takeaways or ["Information not available from current sources."])
    page, y = _write_subheading(doc, page, y, "Recommended Actions")
    actions = [
        _safe(item)
        for item in recommended_actions
        if _safe(item)
    ] if isinstance(recommended_actions, list) else []
    page, y = _write_bullets(doc, page, y, actions or ["No explicit recommendation returned from synthesis model."])
    page, y = _write_subheading(doc, page, y, "Conclusion")
    page, y = _write_paragraph(doc, page, y, conclusion)

    if references:
        page, y = _write_heading(doc, page, y, "6. References", font_size=14)
        for idx, ref in enumerate(references, start=1):
            ref_title = _safe(ref.get("title"), "Untitled source")
            ref_url = _safe(ref.get("url"), "")
            page, y = _write_subheading(doc, page, y, f"[{idx}] {ref_title}", font_size=10.8)
            if ref_url:
                page, y = _write_link(doc, page, y, ref_url, label=ref_url, font_size=10)
            page, y = _write_paragraph(doc, page, y, _safe(ref.get("published_date"), "Unknown publication date"), font_size=10)
            page, y = _write_paragraph(doc, page, y, f"Type: {_safe(ref.get('source_type'), 'unknown')}", font_size=10)
            page, y = _write_paragraph(doc, page, y, f"Domain: {_safe(ref.get('domain'), 'unknown')}", font_size=10)
            y = _draw_divider(page, y)

    for i in range(doc.page_count):
        _write_footer(doc[i], i + 1, doc.page_count)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


def generate_pdf_from_report(report_data: dict, session_id: str) -> str:
    """Backward-compatible helper used by older call sites."""
    report_json_path = f"report_{session_id}.json"
    report_pdf_path = f"report_{session_id}.pdf"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    generate_pdf(report_json_path, report_pdf_path)
    return f"/api/pdf/download/{session_id}"
