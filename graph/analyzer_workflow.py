from langgraph.graph import StateGraph, END
from models.schemas import AnalyzerState
from agents.analyzer_agents import (
    product_extraction_agent,
    competitor_discovery_agent,
    competitor_research_agent,
    comparative_summariser_agent
)

def build_analyzer_graph(checkpointer=None):
    workflow = StateGraph(AnalyzerState)
    
    # Add nodes
    workflow.add_node("extractor", product_extraction_agent)
    workflow.add_node("discoverer", competitor_discovery_agent)
    workflow.add_node("researcher", competitor_research_agent)
    workflow.add_node("summariser", comparative_summariser_agent)
    
    # Define edges
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "discoverer")
    workflow.add_edge("discoverer", "researcher")
    workflow.add_edge("researcher", "summariser")
    workflow.add_edge("summariser", END)
    
    return workflow.compile(checkpointer=checkpointer)
