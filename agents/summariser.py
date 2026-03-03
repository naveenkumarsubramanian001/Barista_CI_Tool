import json
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ResearchState, FinalReport, Insight
from config import get_llm

def summariser_agent(state: ResearchState) -> ResearchState:
    """
    Summarizes the top 3 articles and builds the final report.
    """
    top_articles = state.get("top_articles", [])
    original_query = state.get("original_query", "")
    all_articles = state.get("articles", []) # For references
    
    if not top_articles:
        return state
        
    prompt = ChatPromptTemplate.from_template("""
    Create a research report on "{query}".
    
    Generate summaries for ONLY these top articles:
    {articles_json}
    
    Each summary must:
    - Be 5-8 sentences long.
    - Use ONLY provided content.
    - Include a citation_id mapping to the original index provided in JSON.
    
    You MUST return a valid JSON object with the following structure:
    {{
      "report_title": "string",
      "top_insights": [
        {{
          "title": "string", 
          "brief_summary": "string", 
          "citation_id": integer
        }}
      ]
    }}
    """)
    
    from utils.json_utils import safe_json_extract
    
    articles_data = []
    
    for i, article in enumerate(top_articles):
        articles_data.append({
            "citation_id": i + 1,
            "title": article.title,
            "snippet": article.snippet
        })
        
    chain = prompt | get_llm()
    response = chain.invoke({
        "query": original_query,
        "articles_json": json.dumps(articles_data)
    })
    
    try:
        # Use safe_json_extract for robustness
        data = safe_json_extract(response.content)
        
        # Ensure top_insights exists (fallback if model hallucinated key names)
        if "top_insights" not in data and "insights" in data:
            data["top_insights"] = data.pop("insights")
            
        # Add references to the data for schemas
        data["references"] = [a.model_dump() for a in all_articles]
        
        # Validate with Pydantic
        report = FinalReport(**data)
        state["final_report"] = report.model_dump()
    except Exception as e:
        state["error"] = f"Summariser failed: {str(e)}"
        print(f"Summariser Debug - Raw Response: {response.content}")
        
    return state
