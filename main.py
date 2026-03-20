import json
import asyncio
from langgraph.checkpoint.memory import MemorySaver
from graph.workflow import build_graph


async def main():
    query = "tell me about openai's chatgpt and their new features"

    # Initialize checkpointer for saving graph state
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
        "retry_counts": {"decomposer": 0, "search": 0, "summariser": 0},
        "error": None,
        "search_days_used": None,
        "selected_articles": [],
    }

    print("\n🚀 Starting LangGraph Workflow Phase 1 (Search & Filter)...")
    config = {"configurable": {"thread_id": "1"}}

    # Phase 1: Run until the interrupt before "summariser"
    await app.ainvoke(initial_state, config=config)

    print("\n⏳ Workflow paused. Waiting for human-in-the-loop selection...")

    # Retrieve the state where the graph paused
    current_state = app.get_state(config)
    state_values = current_state.values

    final_ranked = state_values.get("final_ranked_output", {})
    official_top = final_ranked.get("official_sources", [])
    trusted_top = final_ranked.get("trusted_sources", [])

    print(
        f"\n[Frontend Mock] Found {len(official_top)} Official and {len(trusted_top)} Trusted articles."
    )

    # Simulate Frontend selection: Mocking that the user selected all the top articles
    selected_urls = [a.url for a in official_top] + [a.url for a in trusted_top]

    print(
        f"[Frontend Mock] User selected {len(selected_urls)} articles. Resuming workflow..."
    )

    # Update the graph state with the user's selection
    app.update_state(config, {"selected_articles": selected_urls})

    # Phase 2: Resume workflow from the interrupt point
    print("\n🚀 Starting LangGraph Workflow Phase 2 (Summarise & PDF)...")
    final_output = await app.ainvoke(None, config=config)

    if final_output.get("final_report"):
        print("\n--- FINAL RESEARCH REPORT ---\n")
        print(json.dumps(final_output["final_report"], indent=2))

        with open("report.json", "w") as f:
            json.dump(final_output["final_report"], f, indent=2)
            print("\nReport saved to report.json")

        # Generate PDF report with citations
        from utils.pdf_report import generate_pdf

        generate_pdf("report.json", "report.pdf")
    else:
        print("\n❌ Failed to generate report.")
        if final_output.get("error"):
            print(f"Error: {final_output['error']}")
        # Generate a graceful no-data PDF
        from utils.pdf_report import generate_no_data_pdf

        generate_no_data_pdf(query, "report.pdf")


if __name__ == "__main__":
    asyncio.run(main())
