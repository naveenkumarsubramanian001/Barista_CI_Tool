import json
from pathlib import Path

import pytest


def _generator():
    pytest.importorskip("fpdf")
    from utils.comparative_pdf_report import generate_comparative_pdf

    return generate_comparative_pdf


@pytest.mark.parametrize(
    "payload",
    [
        {
            "report_title": "Comparative Intelligence Report",
            "executive_summary": "This is an executive summary.",
            "competitors": [
                {
                    "name": "Competitor A",
                    "domain": "a.example.com",
                    "strengths": ["Brand"],
                    "weaknesses": ["Pricing"],
                    "pricing_strategy": "Premium",
                    "key_features": ["Feature A1"],
                }
            ],
            "user_product_positioning": "Mid-market value proposition.",
            "recommendations": ["Improve onboarding"],
        },
        {
            "report_title": "Unicode ✅ 测试",
            "executive_summary": "Résumé with emojis 🚀 and multilingual text العربية.",
            "competitors": [],
            "recommendations": ["Handle unicode robustly."],
        },
        {
            # Missing fields regression path
        },
    ],
)
def test_generate_comparative_pdf_valid_outputs(tmp_path, payload):
    generate_comparative_pdf = _generator()
    report_path = tmp_path / "report.json"
    pdf_path = tmp_path / "report.pdf"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    generate_comparative_pdf(str(report_path), str(pdf_path))

    assert pdf_path.exists()
    content = pdf_path.read_bytes()
    assert content.startswith(b"%PDF")
    assert len(content) > 200


def test_generate_comparative_pdf_malformed_json_fallback(tmp_path):
    generate_comparative_pdf = _generator()
    report_path = tmp_path / "broken.json"
    pdf_path = tmp_path / "report.pdf"
    report_path.write_text("{ invalid json", encoding="utf-8")

    generate_comparative_pdf(str(report_path), str(pdf_path))

    assert pdf_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF")


def test_generated_pdf_contains_expected_sections(tmp_path):
    generate_comparative_pdf = _generator()
    pypdf = pytest.importorskip("pypdf")

    report_path = tmp_path / "report.json"
    pdf_path = tmp_path / "report.pdf"

    payload = {
        "report_title": "AI Competitive Report",
        "executive_summary": "Summary body",
        "competitors": [
            {
                "name": "Competitor Z",
                "domain": "z.example.com",
                "strengths": ["Strength 1"],
                "weaknesses": ["Weakness 1"],
                "pricing_strategy": "Value",
                "key_features": ["Feature 1"],
            }
        ],
        "user_product_positioning": "Positioning text",
        "recommendations": ["Recommendation 1"],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    generate_comparative_pdf(str(report_path), str(pdf_path))

    reader = pypdf.PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)

    assert "Executive Summary" in text
    assert "Competitor 1: Competitor Z" in text
    assert "Recommendations" in text
