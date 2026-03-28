from langgraph.graph import StateGraph, END, START
from models.schemas import ResearchState
from agents.QueryDecomposer import decomposer_agent
from agents.multi_search_agent import multi_search_agent
from agents.summariser import summariser_agent
from agents.discriminators import (
    decomposer_discriminator,
    search_discriminator,
    summariser_discriminator,
)
from nodes.rank_filter import rank_filter_node
from utils.geturl import url_discovery


def build_graph(checkpointer=None):
    workflow = StateGraph(ResearchState)

    # Add Nodes
    workflow.add_node("decomposer", decomposer_agent)
    workflow.add_node("decomposer_validator", decomposer_discriminator)
    workflow.add_node("url_discovery", url_discovery)
    workflow.add_node("search", multi_search_agent)
    workflow.add_node("search_validator", search_discriminator)
    workflow.add_node("ranker", rank_filter_node)
    workflow.add_node("summariser", summariser_agent)
    workflow.add_node("summariser_validator", summariser_discriminator)

    # Add Edges
    workflow.add_edge(START, "decomposer")
    workflow.add_edge("decomposer", "decomposer_validator")

    # Conditional Edges for Retries
    def after_decomposer(state: ResearchState):
        if state.get("validation_feedback") == "APPROVED":
            return "url_discovery"
        if state["retry_counts"]["decomposer"] >= 2:
            return END  # Should ideally be an error state
        return "decomposer"

    workflow.add_conditional_edges("decomposer_validator", after_decomposer)

    workflow.add_edge("url_discovery", "search")

    workflow.add_edge("search", "search_validator")

    def after_search(state: ResearchState):
        if state.get("validation_feedback") == "APPROVED":
            return "ranker"
        if state["retry_counts"]["search"] >= 2:
            return "ranker"  # Allow partial result as per reqs
        return "search"

    workflow.add_conditional_edges("search_validator", after_search)

    workflow.add_edge("ranker", "summariser")
    workflow.add_edge("summariser", "summariser_validator")

    def after_summariser(state: ResearchState):
        if state.get("validation_feedback") == "APPROVED":
            return END
        if state["retry_counts"]["summariser"] >= 2:
            return END
        return "summariser"

    workflow.add_conditional_edges("summariser_validator", after_summariser)

    return workflow.compile(checkpointer=checkpointer, interrupt_before=["summariser"])
