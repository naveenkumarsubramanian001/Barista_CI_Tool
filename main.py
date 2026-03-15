import json
from graph.workflow import build_graph

import json
import asyncio
from graph.workflow import build_graph

async def main():
    query = "tell me about openai's chatgpt and their new features"
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
        "retry_counts": {
            "decomposer": 0,
            "search": 0,
            "summariser": 0
        },
        "error": None
    }
    
    print("\n🚀 Starting LangGraph Workflow (Async enabled)...")
    
    # Execute the workflow
    config = {"configurable": {"thread_id": "1"}}
    final_output = await app.ainvoke(initial_state, config=config)
    
    if final_output.get("final_report"):
        print("\n--- FINAL RESEARCH REPORT ---\n")
        print(json.dumps(final_output["final_report"], indent=2))
        
        with open("report.json", "w") as f:
            json.dump(final_output["final_report"], f, indent=2)
            print("\nReport saved to report.json")
    else:
        print("\n❌ Failed to generate report.")
        if final_output.get("error"):
            print(f"Error: {final_output['error']}")

if __name__ == "__main__":
    asyncio.run(main())
