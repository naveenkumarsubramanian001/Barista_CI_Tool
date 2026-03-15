"""
PDF Report Generator for Barista CI Tool.

Produces a professional, R&D-grade competitive intelligence PDF from report.json.
Features:
  - In-depth introduction referencing all sources with clickable hyperlinks
  - Top 3 insights per category (official & trusted) with citations
  - Citation line directly below each insight title
  - Detailed conclusion synthesizing both source categories
  - Full numbered reference list with clickable URLs
"""

import json
import os
from datetime import datetime
from fpdf import FPDF

try:
    from tavily import TavilyClient
    from config import TAVILY_API_KEY
    _tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
except Exception:
    _tavily_client = None


# ── Unicode → latin-1 sanitisation ─────────────────────────────────
_UNICODE_MAP = {
    "\u2014": "--",    # em-dash
    "\u2013": "-",     # en-dash
    "\u2018": "'",     # left single quote
    "\u2019": "'",     # right single quote
    "\u201c": '"',     # left double quote
    "\u201d": '"',     # right double quote
    "\u2026": "...",   # ellipsis
    "\u2022": "-",     # bullet
    "\u2060": "",      # word joiner (invisible)
    "\u00a0": " ",     # non-breaking space
    "\u200b": "",      # zero-width space
    "\ufeff": "",      # BOM
}

def _sanitize(text: str) -> str:
    """Replace common Unicode chars with latin-1 safe equivalents."""
    if not text:
        return ""
    for uc, repl in _UNICODE_MAP.items():
        text = text.replace(uc, repl)
    # Catch anything else outside latin-1 range
    return text.encode("latin-1", errors="replace").decode("latin-1")


# ── Colour palette (professional / muted corporate) ────────────────
DARK       = (25, 25, 30)
ACCENT     = (0, 82, 165)       # Deep corporate blue
ACCENT_LT  = (0, 110, 200)      # Lighter link blue
SECTION_BG = (235, 240, 248)    # Section heading background
CARD_BG    = (245, 247, 252)    # Insight card background
MUTED      = (90, 95, 105)      # Secondary / meta text
WHITE      = (255, 255, 255)
BORDER_CLR = (200, 208, 220)    # Card border
CITE_CLR   = (160, 50, 50)      # Citation highlight (dark red)


class ReportPDF(FPDF):
    """Professional CI report PDF with header, footer, and styled sections."""

    def __init__(self, title: str = "Barista CI Report"):
        super().__init__()
        self.report_title = title
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(left=18, top=18, right=18)

    # ── Header (pages 2+) ──────────────────────────────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 6, _sanitize(f"CONFIDENTIAL  |  {self.report_title}"), align="L")
        date_str = datetime.now().strftime("%B %d, %Y")
        self.cell(0, 6, date_str, align="R", new_x="LMARGIN", new_y="NEXT")
        # thin rule
        self.set_draw_color(*BORDER_CLR)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    # ── Footer ─────────────────────────────────────────────────────
    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*BORDER_CLR)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(0, 8, "Barista Competitive Intelligence Tool", align="L")
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="R")

    # ── Section heading ────────────────────────────────────────────
    def section_heading(self, number: str, text: str):
        """Numbered section heading with accent bar."""
        self.ln(3)
        # accent bar
        self.set_fill_color(*ACCENT)
        self.rect(self.l_margin, self.get_y(), 3, 8, style="F")
        self.set_x(self.l_margin + 6)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*ACCENT)
        self.cell(0, 8, f"{number}   {text}", new_x="LMARGIN", new_y="NEXT")
        # underline
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.5)
        y = self.get_y() + 1
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(6)

    # ── Sub-heading ────────────────────────────────────────────────
    def sub_heading(self, text: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*DARK)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    # ── Body text ──────────────────────────────────────────────────
    def body_text(self, text: str):
        self.set_font("Helvetica", "", 9.8)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.2, _sanitize(text))
        self.ln(2)

    # ── Body text with inline links ────────────────────────────────
    def body_text_with_links(self, text: str, link_map: dict):
        """
        Render body text. Wherever a reference title (from link_map keys) appears
        in the text, render it as a clickable hyperlink followed by its citation number.
        link_map: { "title string": {"url": "...", "cite_num": N}, ... }
        """
        self.set_font("Helvetica", "", 9.8)
        self.set_text_color(*DARK)

        # Build segments: split text around known reference titles
        segments = []
        remaining = text
        for title_key, info in link_map.items():
            parts = remaining.split(title_key, 1)
            if len(parts) == 2:
                if parts[0]:
                    segments.append(("text", parts[0]))
                segments.append(("link", title_key, info["url"], info["cite_num"]))
                remaining = parts[1]
            # else: title not in remaining, skip
        if remaining:
            segments.append(("text", remaining))

        # If no links were inserted, just render plain
        if not any(s[0] == "link" for s in segments):
            self.multi_cell(0, 5.2, _sanitize(text))
            self.ln(2)
            return

        # Render segments
        for seg in segments:
            if seg[0] == "text":
                self.set_font("Helvetica", "", 9.8)
                self.set_text_color(*DARK)
                self.write(5.2, _sanitize(seg[1]))
            elif seg[0] == "link":
                _, link_text, url, cite_num = seg
                self.set_font("Helvetica", "U", 9.8)
                self.set_text_color(*ACCENT_LT)
                self.write(5.2, _sanitize(link_text), url)
                # citation superscript-style
                self.set_font("Helvetica", "B", 7.5)
                self.set_text_color(*CITE_CLR)
                self.write(5.2, f" [{cite_num}]")
        self.ln(8)

    # ── Insight card ───────────────────────────────────────────────
    def insight_card(self, idx: int, title: str, summary: str,
                     citation_num: int, ref_title: str, ref_url: str,
                     ref_date: str):
        """
        Render a single insight as a professional card:
          - Numbered title
          - Citation line with source title, hyperlink, and date
          - Detailed summary
        """
        usable_w = self.w - self.l_margin - self.r_margin

        # Check if we need a page break (estimate ~45mm per card)
        if self.get_y() + 45 > self.h - 25:
            self.add_page()

        y_top = self.get_y()

        # Card background
        self.set_fill_color(*CARD_BG)
        self.set_draw_color(*BORDER_CLR)

        # ─ Title row ─
        self.set_x(self.l_margin + 2)
        self.set_font("Helvetica", "B", 10.5)
        self.set_text_color(*DARK)
        title_text = _sanitize(f"{idx}. {title}")
        self.multi_cell(usable_w - 4, 6, title_text,
                        new_x="LMARGIN", new_y="NEXT", fill=True)

        # ─ Citation line ─
        self.set_x(self.l_margin + 8)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*CITE_CLR)
        self.write(4.5, f"[{citation_num}] ")
        self.set_text_color(*MUTED)
        self.write(4.5, f"Source: ")
        self.set_font("Helvetica", "UI", 8)
        self.set_text_color(*ACCENT_LT)
        # Truncate long ref titles for the citation line
        display_ref = ref_title if len(ref_title) <= 70 else ref_title[:67] + "..."
        self.write(4.5, _sanitize(display_ref), ref_url)
        if ref_date:
            self.set_font("Helvetica", "I", 7.5)
            self.set_text_color(*MUTED)
            self.write(4.5, _sanitize(f"  ({ref_date})"))
        self.ln(7)

        # ─ Summary ─
        self.set_x(self.l_margin + 8)
        self.set_font("Helvetica", "", 9.2)
        self.set_text_color(*DARK)
        self.multi_cell(usable_w - 16, 5, _sanitize(summary), new_x="LMARGIN", new_y="NEXT")

        y_bottom = self.get_y() + 2

        # Draw border around entire card
        self.set_draw_color(*BORDER_CLR)
        self.set_line_width(0.3)
        self.rect(self.l_margin, y_top - 1, usable_w, y_bottom - y_top + 2)

        # Left accent stripe
        self.set_fill_color(*ACCENT)
        self.rect(self.l_margin, y_top - 1, 2, y_bottom - y_top + 2, style="F")

        self.set_y(y_bottom + 5)


# ── Tavily context fetcher ─────────────────────────────────────────

def _fetch_company_context(report_title: str) -> str:
    """
    Use Tavily to fetch a brief background on the company or product
    mentioned in the report title. Returns a 2-3 sentence summary.
    """
    if not _tavily_client:
        return ""
    try:
        query = f"What is {report_title}? Brief company or product overview"
        response = _tavily_client.search(
            query=query,
            search_depth="basic",
            topic="general",
            max_results=3,
        )
        results = response.get("results", [])
        if results:
            # Take the best snippet and clean it up
            snippets = []
            for r in results[:2]:
                snip = r.get("content", "")
                # Take first 2-3 meaningful sentences
                sentences = [s.strip() for s in snip.split(".") if len(s.strip()) > 30][:3]
                if sentences:
                    snippets.append(". ".join(sentences) + ".")
            return " ".join(snippets)
    except Exception as e:
        print(f"   [PDF] Tavily context fetch skipped: {e}")
    return ""


def _build_intro(report_title, official, trusted, references, company_context=""):
    """Build an in-depth introduction combining company/product context
    with insights from both official and trusted sources. Returns (text, link_map)."""

    n_official = len(official)
    n_trusted  = len(trusted)
    n_total    = n_official + n_trusted
    n_refs     = len(references)

    # Collect unique domains
    domains = sorted({r.get("domain", "unknown") for r in references})
    domain_str = ", ".join(domains) if domains else "various sources"

    # Start with company/product context if available
    intro = ""
    if company_context:
        intro += _sanitize(company_context) + "\n\n"

    intro += (
        f"This competitive intelligence report presents a curated analysis "
        f"on the topic \"{report_title}\". Through systematic web research and "
        f"automated source classification, {n_refs} reference(s) were identified "
        f"and evaluated across the following domain(s): {domain_str}. "
        f"\n\n"
        f"The research yielded {n_total} actionable insight(s) in total: "
        f"{n_official} from official (first-party) sources and {n_trusted} from "
        f"trusted (high-credibility) sources. "
    )

    # Add specific mentions of the insights
    if official:
        off_titles = [f'\"{i.get("title", "")}\"' for i in official]
        intro += (
            f"From official channels, the analysis identified findings related to "
            f"{', '.join(off_titles)}. "
        )
    if trusted:
        tr_titles = [f'\"{i.get("title", "")}\"' for i in trusted]
        intro += (
            f"From trusted third-party coverage, key findings cover "
            f"{', '.join(tr_titles)}. "
        )

    intro += (
        f"\n\n"
        f"Each insight below is accompanied by a direct citation linking to its "
        f"original source. The full reference list with URLs is provided at the "
        f"end of this document for independent verification and further reading. "
        f"This report is intended for internal R&D and strategic planning use."
    )

    # Build link_map: map reference titles to URL + citation number
    link_map = {}
    for i, ref in enumerate(references):
        ref_title = ref.get("title", "")
        if ref_title:
            link_map[ref_title] = {
                "url": ref.get("url", ""),
                "cite_num": i + 1,
            }

    return intro, link_map


def _build_conclusion(report_title, official, trusted, references, company_context=""):
    """Build a detailed conclusion synthesizing both source categories
    alongside company/product context."""
    n_official = len(official)
    n_trusted  = len(trusted)
    n_refs     = len(references)

    conclusion = (
        f"This report analyzed {n_refs} source(s) to extract competitive "
        f"intelligence on \"{report_title}\". "
    )

    if official:
        off_themes = "; ".join(
            f'\"{i.get("title", "")}\"' for i in official
        )
        conclusion += (
            f"Official source intelligence ({n_official} insight(s)) covered: "
            f"{off_themes}. These represent first-party disclosures and product "
            f"announcements that carry the highest reliability weight for "
            f"strategic decision-making. "
        )

    if trusted:
        tr_themes = "; ".join(
            f'\"{i.get("title", "")}\"' for i in trusted
        )
        conclusion += (
            f"Trusted source intelligence ({n_trusted} insight(s)) covered: "
            f"{tr_themes}. These findings come from high-credibility third-party "
            f"outlets and provide broader market context and independent "
            f"validation of the official disclosures. "
        )

    # Date range
    dates = []
    for ref in references:
        d = ref.get("published_date", "")
        if d:
            dates.append(d)
    if dates:
        conclusion += (
            f"\n\n"
            f"The sources span publication dates from {dates[-1].split(',')[0].strip()} "
            f"to {dates[0].split(',')[0].strip()}, providing a timely cross-section "
            f"of recent developments. "
        )

    # Weave in company context for strategic framing
    if company_context:
        conclusion += (
            f"\n\n"
            f"For strategic context: {_sanitize(company_context)} "
        )

    conclusion += (
        f"\n\n"
        f"Recommendation: R&D teams should cross-reference these findings with "
        f"internal roadmap priorities and evaluate competitive positioning "
        f"accordingly. All citations are provided for independent verification."
    )

    return conclusion


def generate_pdf(json_path: str, output_path: str = "report.pdf"):
    """
    Read *json_path* and produce a professional R&D-grade PDF at *output_path*.

    PDF layout
    ----------
    1. Cover / Title page
    2. Introduction (detailed, with inline hyperlinks to sources)
    3. Top Insights — Official Sources (up to 3, each with citation line)
    4. Top Insights — Trusted Sources (up to 3, each with citation line)
    5. Conclusion (synthesising both categories)
    6. References (full numbered list with clickable URLs)
    """
    # ── Load data ───────────────────────────────────────────────────
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    report_title = data.get("report_title", "Competitive Intelligence Report")
    official     = data.get("official_insights", [])[:3]
    trusted      = data.get("trusted_insights", [])[:3]
    references   = data.get("references", [])

    # ── Fetch company/product context via Tavily ───────────────────
    print("   [PDF] Fetching company/product context via Tavily...")
    company_context = _fetch_company_context(report_title)

    # ── Create PDF ──────────────────────────────────────────────────
    pdf = ReportPDF(title=report_title)
    pdf.alias_nb_pages()

    # ================================================================
    # 1. COVER PAGE
    # ================================================================
    pdf.add_page()
    pdf.ln(50)

    # Accent rule
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(1.2)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(8)

    # Title
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*ACCENT)
    pdf.multi_cell(0, 13, _sanitize(report_title), align="C")
    pdf.ln(4)

    # Subtitle
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 7, "Competitive Intelligence Report", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Accent rule
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(1.2)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(12)

    # Metadata block
    date_str = datetime.now().strftime("%B %d, %Y  |  %H:%M")
    meta_lines = [
        f"Generated: {date_str}",
        f"Total Sources Analysed: {len(references)}",
        f"Official Insights: {len(official)}  |  Trusted Insights: {len(trusted)}",
        "Classification: INTERNAL USE -- R&D",
    ]
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTED)
    for line in meta_lines:
        pdf.cell(0, 6.5, _sanitize(line), align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(25)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, "Produced by Barista Competitive Intelligence Tool",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # ================================================================
    # 2. INTRODUCTION
    # ================================================================
    pdf.add_page()
    pdf.section_heading("01", "Introduction")

    intro_text, link_map = _build_intro(report_title, official, trusted, references, company_context)
    pdf.body_text_with_links(intro_text, link_map)

    # ================================================================
    # 3. OFFICIAL SOURCE INSIGHTS
    # ================================================================
    pdf.section_heading("02", "Top Insights from Official Sources")

    if official:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(0, 5, (
            "Official sources are first-party channels operated directly by the "
            "subject entity (e.g., company blogs, press releases, investor pages). "
            "These carry the highest credibility weight."
        ))
        pdf.ln(5)

        for i, insight in enumerate(official, 1):
            cid = insight.get("citation_id", 0)
            ref = references[cid] if cid < len(references) else {}
            pdf.insight_card(
                idx=i,
                title=insight.get("title", "Untitled"),
                summary=insight.get("brief_summary", "No summary available."),
                citation_num=cid + 1,
                ref_title=ref.get("title", ""),
                ref_url=ref.get("url", ""),
                ref_date=ref.get("published_date", ""),
            )
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(0, 6, (
            "No official source insights were identified for this query. "
            "This may indicate limited first-party disclosure on the topic "
            "during the analysed time period."
        ))
        pdf.ln(4)

    # ================================================================
    # 4. TRUSTED SOURCE INSIGHTS
    # ================================================================
    pdf.section_heading("03", "Top Insights from Trusted Sources")

    if trusted:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(0, 5, (
            "Trusted sources are high-credibility third-party outlets such as "
            "major technology publications, industry analysts, and established "
            "news organisations that provide independent coverage and analysis."
        ))
        pdf.ln(5)

        for i, insight in enumerate(trusted, 1):
            cid = insight.get("citation_id", 0)
            ref = references[cid] if cid < len(references) else {}
            pdf.insight_card(
                idx=i,
                title=insight.get("title", "Untitled"),
                summary=insight.get("brief_summary", "No summary available."),
                citation_num=cid + 1,
                ref_title=ref.get("title", ""),
                ref_url=ref.get("url", ""),
                ref_date=ref.get("published_date", ""),
            )
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*MUTED)
        pdf.multi_cell(0, 6, (
            "No trusted source insights were identified for this query. "
            "This may indicate limited third-party coverage on the topic "
            "during the analysed time period."
        ))
        pdf.ln(4)

    # ================================================================
    # 5. CONCLUSION
    # ================================================================
    pdf.section_heading("04", "Conclusion & Recommendations")

    conclusion_text = _build_conclusion(report_title, official, trusted, references, company_context)
    pdf.body_text(conclusion_text)

    # ================================================================
    # 6. REFERENCES
    # ================================================================
    pdf.add_page()
    pdf.section_heading("05", "References")

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5, (
        "Full reference list for all sources cited in this report. "
        "URLs are clickable for independent verification."
    ))
    pdf.ln(5)

    if references:
        usable_w = pdf.w - pdf.l_margin - pdf.r_margin
        for i, ref in enumerate(references):
            # Page break check
            if pdf.get_y() + 25 > pdf.h - 25:
                pdf.add_page()

            cite_num = i + 1
            ref_title = ref.get("title", "Untitled")
            url       = ref.get("url", "")
            pub_date  = ref.get("published_date", "N/A")
            src_type  = ref.get("source_type", "unknown").capitalize()
            domain    = ref.get("domain", "")

            # Citation number + title
            pdf.set_font("Helvetica", "B", 9.5)
            pdf.set_text_color(*DARK)
            pdf.multi_cell(0, 5.5, _sanitize(f"[{cite_num}]  {ref_title}"),
                           new_x="LMARGIN", new_y="NEXT")

            # URL (clickable)
            pdf.set_x(pdf.l_margin + 10)
            pdf.set_font("Helvetica", "U", 8.5)
            pdf.set_text_color(*ACCENT_LT)
            # Truncate very long URLs for display
            display_url = url if len(url) <= 90 else url[:87] + "..."
            pdf.cell(0, 5, display_url, new_x="LMARGIN", new_y="NEXT", link=url)

            # Meta line
            pdf.set_x(pdf.l_margin + 10)
            pdf.set_font("Helvetica", "I", 7.5)
            pdf.set_text_color(*MUTED)
            meta = f"Published: {pub_date}  |  Type: {src_type}  |  Domain: {domain}"
            pdf.cell(0, 5, _sanitize(meta), new_x="LMARGIN", new_y="NEXT")

            # Separator
            pdf.ln(2)
            pdf.set_draw_color(*BORDER_CLR)
            pdf.set_line_width(0.2)
            pdf.line(pdf.l_margin + 5, pdf.get_y(),
                     pdf.w - pdf.r_margin - 5, pdf.get_y())
            pdf.ln(4)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 8, "No references available.",
                 new_x="LMARGIN", new_y="NEXT")

    # ── Save ────────────────────────────────────────────────────────
    pdf.output(output_path)
    print(f"\n📄 PDF report saved to {output_path}")


# Allow standalone usage:  python utils/pdf_report.py
if __name__ == "__main__":
    generate_pdf("report.json", "report.pdf")


def generate_no_data_pdf(query: str, output_path: str = "report.pdf"):
    """
    Generate a graceful 'no data found' PDF when the pipeline finds no articles.
    Includes the query, search parameters, and suggestions.
    """
    pdf = ReportPDF(title="No Data Found")
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.ln(40)

    # Title
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(1.2)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 12, "No Relevant Articles Found", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(1.2)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(12)

    # Query details
    date_str = datetime.now().strftime("%B %d, %Y  |  %H:%M")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6.5, _sanitize(f"Query: {query}"), align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6.5, f"Generated: {date_str}", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6.5, "Search windows tried: 30, 90, 180 days", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)

    # Explanation
    pdf.section_heading("01", "Summary")
    pdf.body_text(
        "The Barista Competitive Intelligence Tool was unable to find any "
        "relevant articles for the given query within the search parameters. "
        "Multiple time windows (30, 90, and 180 days) were attempted before "
        "determining that no qualifying content is available.\n\n"
        "This may occur due to:\n"
        "- The topic is too niche or specific for the configured source domains.\n"
        "- No recent publications exist on this topic within the time window.\n"
        "- The company or product domains are not correctly configured.\n"
        "- The search query may need to be rephrased for better coverage."
    )

    pdf.section_heading("02", "Recommendations")
    pdf.body_text(
        "1. Try broadening the query with more general terms.\n"
        "2. Specify a wider time range in your query (e.g., 'last 6 months' or 'past year').\n"
        "3. Verify that the company domains and trusted domains are correctly configured.\n"
        "4. Check that the Tavily API key is valid and has remaining search credits.\n"
        "5. Consider adding more trusted source domains to expand coverage."
    )

    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, "Produced by Barista Competitive Intelligence Tool",
             align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.output(output_path)
    print(f"\n📄 No-data PDF report saved to {output_path}")
