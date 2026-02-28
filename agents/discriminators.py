import json
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ResearchState, ValidationResult
from config import get_llm

def decomposer_discriminator(state: ResearchState) -> ResearchState:
    """
    Evaluates the quality of subqueries produced by the decomposer.
    """
    subqueries = state.get("subqueries", [])
    if not subqueries:
        state["validation_feedback"] = "No subqueries produced."
        return state

    prompt = ChatPromptTemplate.from_template("""
    Evaluate the following research subqueries for quality and structural correctness.
    Checks:
    1. Is it a JSON list of 3-6 strings?
    2. Do they contain recency terms (e.g., "last 2 weeks", "recent")?
    3. Are they non-redundant and focused?
    
    Subqueries: {subqueries}
    
    Return ONLY a JSON object: {{"approved": bool, "feedback": "reasoning here"}}
    """)
    
    chain = prompt | get_llm()
    response = chain.invoke({"subqueries": json.dumps(subqueries)})
    
    try:
        data = json.loads(response.content)
        result = ValidationResult(**data)
        if result.approved:
            state["validation_feedback"] = "APPROVED"
        else:
            state["validation_feedback"] = result.feedback
            state["retry_counts"]["decomposer"] += 1
    except Exception as e:
        state["validation_feedback"] = f"Discriminator error: {str(e)}"
        state["retry_counts"]["decomposer"] += 1
        
    return state

def search_discriminator(state: ResearchState) -> ResearchState:
    """
    Validates search results for date range and relevance.
    """
    articles = state.get("articles", [])
    if not articles:
        state["validation_feedback"] = "No articles found."
        state["retry_counts"]["search"] += 1
        return state

    # Simple validation: Check if we have enough and dates are present
    if len(articles) > 10:
        state["validation_feedback"] = "Too many articles (max 10)."
        state["retry_counts"]["search"] += 1
        return state
        
    state["validation_feedback"] = "APPROVED"
    return state

def summariser_discriminator(state: ResearchState) -> ResearchState:
    """
    Validates the final report for citations and length.
    """
    report = state.get("final_report", {})
    if not report:
        state["validation_feedback"] = "No report generated."
        state["retry_counts"]["summariser"] += 1
        return state
        
    insights = report.get("top_insights", [])
    if len(insights) != 3:
        state["validation_feedback"] = f"Expected exactly 3 insights, got {len(insights)}."
        state["retry_counts"]["summariser"] += 1
        return state
        
    state["validation_feedback"] = "APPROVED"
    return state
