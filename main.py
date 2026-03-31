"""
End-to-End Pipeline Test Script
Run this file directly to test the entire LangGraph workflow 
(including search, filtering, and report generation) from the command line.
"""

import json
import asyncio
import os
import argparse
from graph.workflow import build_graph
from langgraph.checkpoint.memory import MemorySaver
from utils.logger import (
    console, banner, section, info, success, warning, error,
    report_summary, phase_progress
)
from rich.panel import Panel
from rich import box
from utils.pdf_report import generate_pdf


async def main():
    parser = argparse.ArgumentParser(description="Test the Barista CI Pipeline")
    parser.add_argument("--query", type=str, default="openais chatgpt new features and updates ", help="The research query")
    args = parser.parse_args()

    query = args.query
    banner("🚀 BARISTA CI TOOL [TEST MODE]", f"Query: {query}", style="bold magenta")
    
    # Needs a memory saver for interrupt_before logic to work properly
    checkpointer = MemorySaver()
    app = build_graph(checkpointer=checkpointer)

    initial_state = {
        "original_query": query,
        "subqueries": [],
        "official_sources": [],
        "trusted_sources": [],
        "final_ranked_output": {},
        "final_report": None,
        "company_domains": [],
        "trusted_domains": [],
        "validation_feedback": "",
        "validation_passed": False,
        "validation_metrics": {},
        "decomposition_score": 0.0,
        "redundancy_pairs": [],
        "coverage_gaps": [],
        "semantic_warnings": [],
        "retry_counts": {
            "decomposer": 0,
            "search": 0,
            "summariser": 0,
            "report_guardrail": 0
        },
        "error": None,
        "search_days_used": None,
        "selected_articles": [],
        "logs": [],
        "stages": [],
        "current_stage": "understand",
        "progress_percentage": 5,
        "guardrail_status": "unchecked",
        "guardrail_reason": "",
        "guardrail_blocked": False,
        "primary_entity": "",
    }

    config = {"configurable": {"thread_id": "test_session_1"}}

    # Phase 1: Search and Filter
    with phase_progress("Running Phase 1: Search & Filter Pipeline"):
        try:
            # Note: intercept_before stops the graph right before it enters summariser
            state_output = await app.ainvoke(initial_state, config=config, interrupt_before=["summariser"])
        except TypeError:
            # Safe fallback if ainvoke doesn't like interrupt_before
            state_output = await app.ainvoke(initial_state, config=config)

    # If blocked by query guardrail early
    if state_output.get("guardrail_blocked"):
        error(f"Query was blocked by Guardrail: {state_output.get('guardrail_reason')}")
        return

    # Simulate Human-in-the-loop: select top 3 articles
    section("Simulating Human Review", "👀")
    final_ranked = state_output.get("final_ranked_output", {})
    official = final_ranked.get("official_sources", [])
    trusted = final_ranked.get("trusted_sources", [])
    
    selected_urls = []
    
    info("\nTop Official Sources:")
    for i, src in enumerate(official[:2]):
        url = src.get("url") if isinstance(src, dict) else src.url
        title = src.get("title") if isinstance(src, dict) else src.title
        console.print(f"  [green]✓ {title}[/green] ({url})")
        selected_urls.append(url)

    info("\nTop Trusted Sources:")
    for i, src in enumerate(trusted[:2]):
        url = src.get("url") if isinstance(src, dict) else src.url
        title = src.get("title") if isinstance(src, dict) else src.title
        console.print(f"  [green]✓ {title}[/green] ({url})")
        selected_urls.append(url)

    if not selected_urls:
        error("No sources found to summarize!")
        return

    # Phase 2: Resume matching API behaviour
    with phase_progress("Running Phase 2: Summarisation & Report Guardrail"):
        # We manually update the state with the user selection
        await app.aupdate_state(config, {"selected_articles": selected_urls})
        final_output = await app.ainvoke(None, config=config)

    if final_output.get("guardrail_blocked"):
        error(f"Report was blocked by Output Guardrail: {final_output.get('guardrail_reason')}")
        return

    if final_output.get("final_report"):
        report = final_output["final_report"]
        
        banner("📋 FINAL REPORT GENERATED", report.get("report_title", ""), style="bold green")
        
        if report.get("executive_summary"):
            console.print(Panel(
                report["executive_summary"],
                title="[bold]Executive Summary[/bold]",
                border_style="blue",
                box=box.ROUNDED,
            ))

        # Save JSON
        with open("test_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            success("Report JSON saved to test_report.json")
            
        # Optional: Test PDF generation
        try:
            generate_pdf("test_report.json", "test_report.pdf")
            success("Report PDF saved to test_report.pdf")
        except Exception as e:
            warning(f"Could not generate PDF (maybe missing wkhtmltopdf?): {e}")

    else:
        error("Failed to generate report.")
        if final_output.get("error"):
            error(f"Error: {final_output['error']}")

    banner("✨ PIPELINE COMPLETE", style="bold green")


if __name__ == "__main__":
    asyncio.run(main())
