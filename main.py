import json
import asyncio
from graph.workflow import build_graph
from utils.logger import (
    console, banner, section, info, success, warning, error,
    report_summary, phase_progress
)
from rich.panel import Panel
from rich import box


async def main():
    query = "Samsung new features One Ui 8.5"
    
    banner("🚀 BARISTA CI TOOL", f"Query: {query}", style="bold magenta")
    
    app = build_graph()

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
            "summariser": 0
        },
        "error": None,
        "search_days_used": None,
        "selected_articles": [],
        "logs": [],
    }

    with phase_progress("Running full CI pipeline"):
        config = {"configurable": {"thread_id": "1"}}
        final_output = await app.ainvoke(initial_state, config=config)

    if final_output.get("final_report"):
        report = final_output["final_report"]
        
        banner("📋 FINAL REPORT", report.get("report_title", ""), style="bold green")
        
        # Executive Summary
        if report.get("executive_summary"):
            console.print(Panel(
                report["executive_summary"],
                title="[bold]Executive Summary[/bold]",
                border_style="blue",
                box=box.ROUNDED,
            ))

        # Key Findings
        if report.get("key_findings"):
            section("Key Findings", "🔑")
            for i, kf in enumerate(report["key_findings"], 1):
                console.print(f"\n  [bold cyan][{i}][/bold cyan] [bold]{kf.get('finding_title', '')}[/bold]")
                console.print(f"      {kf.get('finding_summary', '')}")
                console.print(f"      [dim]Sources: {kf.get('source_ids', [])}[/dim]")

        # Cross-Source Analysis
        if report.get("cross_source_analysis"):
            console.print(Panel(
                report["cross_source_analysis"],
                title="[bold]Cross-Source Analysis[/bold]",
                border_style="yellow",
                box=box.ROUNDED,
            ))

        # References
        refs = report.get("references", [])
        if refs:
            section(f"References ({len(refs)})", "📚")
            for i, ref in enumerate(refs, 1):
                title = ref.get("title", "")[:60]
                url = ref.get("url", "")
                console.print(f"  [cyan][{i}][/cyan] {title}")
                console.print(f"       [dim]{url}[/dim]")

        # Save report
        with open("report.json", "w") as f:
            json.dump(report, f, indent=2)
            success("Report saved to report.json")
    else:
        error("Failed to generate report.")
        if final_output.get("error"):
            error(f"Error: {final_output['error']}")

    banner("✨ PIPELINE COMPLETE", style="bold green")


if __name__ == "__main__":
    asyncio.run(main())
