"""Comparative PDF report generator for analyzer workflow."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from fpdf import FPDF


_UNICODE_MAP = {
    "\u2014": "--",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u2022": "-",
    "\u00a0": " ",
    "\u200b": "",
    "\ufeff": "",
}


def _sanitize(text: Any) -> str:
    value = "" if text is None else str(text)
    for old, new in _UNICODE_MAP.items():
        value = value.replace(old, new)
    return value.encode("latin-1", errors="replace").decode("latin-1")


class ComparativePDF(FPDF):
    def __init__(self, title: str):
        super().__init__()
        self.report_title = _sanitize(title)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(16, 16, 16)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 5, self.report_title, align="L")
        self.cell(0, 5, datetime.now().strftime("%Y-%m-%d"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 8, f"Page {self.page_no()}", align="R")


def _section(pdf: ComparativePDF, title: str):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 55, 120)
    pdf.cell(0, 8, _sanitize(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(20, 55, 120)
    pdf.set_line_width(0.3)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(3)


def _paragraph(pdf: ComparativePDF, text: str):
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(25, 25, 25)
    pdf.multi_cell(0, 5.3, _sanitize(text))
    pdf.ln(1)


def _bullet_list(pdf: ComparativePDF, items: List[Any]):
    if not items:
        _paragraph(pdf, "No data available.")
        return
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(25, 25, 25)
    for item in items:
        pdf.multi_cell(0, 5.1, f"- {_sanitize(item)}")
    pdf.ln(1)


def _render_competitor(pdf: ComparativePDF, competitor: Dict[str, Any], idx: int):
    name = competitor.get("name") or f"Competitor {idx}"
    domain = competitor.get("domain") or ""
    _section(pdf, f"Competitor {idx}: {name}")
    if domain:
        _paragraph(pdf, f"Domain: {domain}")

    _paragraph(pdf, "Strengths")
    _bullet_list(pdf, competitor.get("strengths") or [])

    _paragraph(pdf, "Weaknesses")
    _bullet_list(pdf, competitor.get("weaknesses") or [])

    pricing = competitor.get("pricing_strategy")
    if pricing:
        _paragraph(pdf, f"Pricing strategy: {pricing}")

    _paragraph(pdf, "Key features")
    _bullet_list(pdf, competitor.get("key_features") or [])


def generate_comparative_pdf(json_path: str, pdf_path: str):
    """Generate a real comparative intelligence PDF from analyzer report JSON."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        data = {
            "report_title": "Comparative Intelligence Report",
            "executive_summary": "Report data could not be loaded.",
            "competitors": [],
            "recommendations": [f"Data loading error: {exc}"],
        }

    title = data.get("report_title") or "Comparative Intelligence Report"
    pdf = ComparativePDF(title)
    pdf.alias_nb_pages()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(0, 10, _sanitize(title))
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    _section(pdf, "Executive Summary")
    _paragraph(pdf, data.get("executive_summary") or "No summary available.")

    competitors = data.get("competitors") or []
    if isinstance(competitors, list) and competitors:
        for idx, competitor in enumerate(competitors, start=1):
            if isinstance(competitor, dict):
                _render_competitor(pdf, competitor, idx)

    positioning = data.get("user_product_positioning")
    if positioning:
        _section(pdf, "User Product Positioning")
        _paragraph(pdf, positioning)

    _section(pdf, "Recommendations")
    _bullet_list(pdf, data.get("recommendations") or [])

    pdf.output(pdf_path)
