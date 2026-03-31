"""
LangGraph workflow with multi-source search aggregation.
All available search APIs run in parallel, results merged and deduped,
then passed through hybrid fuzzy discriminator.
Now includes Guardrail AI at input (before decomposer) and output
(after summariser) to block malicious queries and validate reports.
"""

import inspect

from langgraph.graph import StateGraph, END, START
from models.schemas import ResearchState
from agents.QueryDecomposer import decomposer_agent
from agents.multi_search_agent import multi_search_agent
from agents.summariser import summariser_agent
from agents.guardrails import query_guardrail, report_guardrail
from agents.discriminators import (
    decomposer_discriminator, 
    search_discriminator, 
    summariser_discriminator
)
from nodes.rank_filter import rank_filter_node
from utils.geturl import url_discovery


STAGE_LABELS = {
    "understand": "Understanding your query",
    "identify": "Identifying topic & domain",
    "collect": "Collecting source candidates",
    "filter": "Filtering & ranking content",
    "analyze": "Analyzing selected documents",
    "prepare": "Preparing results for review",
}


def _default_stages():
    return [
        {"id": key, "label": value, "status": "pending"}
        for key, value in STAGE_LABELS.items()
    ]


def _ensure_tracking(state: ResearchState) -> ResearchState:
    if not state.get("stages"):
        state["stages"] = _default_stages()
    if "logs" not in state:
        state["logs"] = []
    if "current_stage" not in state:
        state["current_stage"] = "initializing"
    if "progress_percentage" not in state:
        state["progress_percentage"] = 0
    return state


def _mark_stage(
    state: ResearchState,
    stage_id: str,
    status: str,
    current_stage: str | None = None,
    progress: int | None = None,
    log_message: str | None = None,
) -> ResearchState:
    state = _ensure_tracking(state)
    for stage in state["stages"]:
        if stage.get("id") == stage_id:
            stage["status"] = status
            break
    if current_stage is not None:
        state["current_stage"] = current_stage
    if progress is not None:
        state["progress_percentage"] = progress
    if log_message:
        state["logs"].append(log_message)
    return state


async def _run_node(fn, state: ResearchState):
    result = fn(state)
    if inspect.isawaitable(result):
        result = await result
    return result


async def query_guardrail_stage(state: ResearchState) -> ResearchState:
    """Guardrail Layer 1: classify query before decomposition."""
    _mark_stage(state, "understand", "running", current_stage="understand", progress=5,
                log_message="Running query safety guardrail...")
    result = await _run_node(query_guardrail, state)
    return result


async def decomposer_stage(state: ResearchState) -> ResearchState:
    _mark_stage(state, "understand", "running", current_stage="understand", progress=12)
    result = await _run_node(decomposer_agent, state)
    _mark_stage(result, "understand", "completed", current_stage="identify", progress=20)
    return result


async def url_discovery_stage(state: ResearchState) -> ResearchState:
    _mark_stage(state, "identify", "running", current_stage="identify", progress=28)
    result = await _run_node(url_discovery, state)
    _mark_stage(result, "identify", "completed", current_stage="collect", progress=36)
    return result


async def search_stage(state: ResearchState) -> ResearchState:
    _mark_stage(state, "collect", "running", current_stage="collect", progress=45)
    result = await _run_node(multi_search_agent, state)
    _mark_stage(result, "collect", "completed", current_stage="filter", progress=58)
    return result


async def search_validator_stage(state: ResearchState) -> ResearchState:
    _mark_stage(state, "filter", "running", current_stage="filter", progress=66)
    result = await _run_node(search_discriminator, state)
    if result.get("validation_feedback") == "APPROVED":
        _mark_stage(result, "filter", "completed", current_stage="analyze", progress=76)
    return result


async def ranker_stage(state: ResearchState) -> ResearchState:
    _mark_stage(state, "analyze", "running", current_stage="analyze", progress=84)
    result = await _run_node(rank_filter_node, state)
    _mark_stage(result, "analyze", "completed", current_stage="prepare", progress=92)
    _mark_stage(
        result,
        "prepare",
        "completed",
        current_stage="select",
        progress=100,
        log_message="Review stage ready. Awaiting human selection.",
    )
    return result


async def summariser_stage(state: ResearchState) -> ResearchState:
    _mark_stage(state, "analyze", "running", current_stage="analyze", progress=94)
    result = await _run_node(summariser_agent, state)
    _mark_stage(result, "analyze", "completed", current_stage="prepare", progress=98)
    return result


async def summariser_validator_stage(state: ResearchState) -> ResearchState:
    _mark_stage(state, "prepare", "running", current_stage="prepare", progress=99)
    result = await _run_node(summariser_discriminator, state)
    if result.get("validation_feedback") == "APPROVED":
        _mark_stage(result, "prepare", "completed", current_stage="finished", progress=100)
    return result


async def report_guardrail_stage(state: ResearchState) -> ResearchState:
    """Guardrail Layer 2: validate report before delivery."""
    _mark_stage(state, "prepare", "running", current_stage="prepare", progress=100,
                log_message="Running report safety guardrail...")
    result = await _run_node(report_guardrail, state)
    return result


def build_graph(checkpointer=None):
    workflow = StateGraph(ResearchState)

    # Add Nodes
    workflow.add_node("query_guardrail", query_guardrail_stage)   # NEW: Layer 1 guardrail
    workflow.add_node("decomposer", decomposer_stage)
    workflow.add_node("decomposer_validator", decomposer_discriminator)
    workflow.add_node("url_discovery", url_discovery_stage)
    workflow.add_node("search", search_stage)  # All providers in parallel
    workflow.add_node("search_validator", search_validator_stage)
    workflow.add_node("ranker", ranker_stage)
    workflow.add_node("summariser", summariser_stage)
    workflow.add_node("summariser_validator", summariser_validator_stage)
    workflow.add_node("report_guardrail", report_guardrail_stage)  # NEW: Layer 2 guardrail

    # --- Edges ---

    # Start → Query Guardrail → (blocked? END : decomposer)
    workflow.add_edge(START, "query_guardrail")

    def after_query_guardrail(state: ResearchState):
        if state.get("guardrail_blocked"):
            return END
        return "decomposer"

    workflow.add_conditional_edges("query_guardrail", after_query_guardrail)

    workflow.add_edge("decomposer", "decomposer_validator")

    # Conditional Edges for Decomposer Retries
    def after_decomposer(state: ResearchState):
        if state.get("validation_feedback") == "APPROVED":
            return "url_discovery"
        if state["retry_counts"]["decomposer"] >= 2:
            return END
        return "decomposer"

    workflow.add_conditional_edges("decomposer_validator", after_decomposer)

    workflow.add_edge("url_discovery", "search")
    workflow.add_edge("search", "search_validator")

    def after_search(state: ResearchState):
        if state.get("validation_feedback") == "APPROVED":
            return "ranker"
        if state["retry_counts"]["search"] >= 2:
            return "ranker"  # Allow partial result
        return "search"

    workflow.add_conditional_edges("search_validator", after_search)

    workflow.add_edge("ranker", "summariser")
    workflow.add_edge("summariser", "summariser_validator")

    def after_summariser(state: ResearchState):
        if state.get("validation_feedback") == "APPROVED":
            return "report_guardrail"  # proceed to Layer 2 guardrail
        if state["retry_counts"]["summariser"] >= 2:
            return "report_guardrail"  # allow partial report through guardrail
        return "summariser"

    workflow.add_conditional_edges("summariser_validator", after_summariser)

    # Report Guardrail → (retry summariser if report cleared, else END)
    def after_report_guardrail(state: ResearchState):
        # report_guardrail clears final_report on first failure to trigger retry
        if state.get("final_report") is None and not state.get("guardrail_blocked"):
            retry_counts = state.get("retry_counts") or {}
            if retry_counts.get("report_guardrail", 0) <= 1:
                return "summariser"
        return END

    workflow.add_conditional_edges("report_guardrail", after_report_guardrail)

    return workflow.compile(checkpointer=checkpointer)
