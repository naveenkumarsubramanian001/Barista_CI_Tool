import json
from graph.workflow import build_graph

def main():
    query = "Im going to launch smartphone for 10000 rupee i want to know about my competetor nothing's advancements and news in this segment since the last 6 months"
    
    app = build_graph()
    
    initial_state = {
        "original_query": query,
        "subqueries": [],
        "articles": [],
        "top_articles": [],
        "final_report": None,
        "validation_feedback": "",
        "retry_counts": {
            "decomposer": 0,
            "search": 0,
            "summariser": 0
        },
        "error": None
    }
    
    print("\n🚀 Starting LangGraph Workflow...")
    
    # Execute the workflow
    # We use invoke to get the final result. For a simple prototype, this is cleaner.
    # If we wanted to see live updates, we'd use stream properly.
    
    config = {"configurable": {"thread_id": "1"}}
    final_output = app.invoke(initial_state, config=config)
    
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
    main()
