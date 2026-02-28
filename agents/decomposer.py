import json
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ResearchState, DecomposedQueries
from config import get_llm

def decomposer_agent(state: ResearchState) -> ResearchState:
    """
    Decomposes the original query into focused subqueries.
    """
    query = state["original_query"]
    feedback = state.get("validation_feedback", "")
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert research assistant. Decompose the following user query into 3-6 focused subqueries.
    Each subquery must:
    - Be specific and focused.
    - Embed time awareness for the past 14 days (e.g., "published in the last 2 weeks", "recent breakthroughs since Feb 2024").
    - Avoid redundancy.
    
    Original Query: {query}
    
    {feedback_section}
    
    Return ONLY a JSON object: {{"subqueries": ["query1", "query2", ...]}}
    """)
    
    feedback_section = f"Previous attempt failed. Feedback: {feedback}" if feedback else ""
    
    chain = prompt | get_llm()
    response = chain.invoke({"query": query, "feedback_section": feedback_section})
    
    try:
        # ChatOllama with format="json" should return JSON string
        data = json.loads(response.content)
        parsed = DecomposedQueries(**data)
        state["subqueries"] = parsed.subqueries
        state["validation_feedback"] = "" # Clear feedback on success
    except Exception as e:
        state["error"] = f"Decomposer failed to produce valid JSON: {str(e)}"
        
    return state
