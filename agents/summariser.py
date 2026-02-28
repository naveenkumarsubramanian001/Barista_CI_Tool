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
    - Include a citation_id mapping to the original index (1, 2, 3).
    
    Return a JSON object:
    {{
      "report_title": "...",
      "top_insights": [
        {{"title": "...", "brief_summary": "...", "citation_id": 1}}
      ]
    }}
    """)
    
    articles_data = []
    url_to_idx = {a.url: i + 1 for i, a in enumerate(all_articles)}
    
    for article in top_articles:
        articles_data.append({
            "title": article.title,
            "snippet": article.snippet,
            "url": article.url
        })
        
    chain = prompt | get_llm()
    response = chain.invoke({
        "query": original_query,
        "articles_json": json.dumps(articles_data)
    })
    
    try:
        data = json.loads(response.content)
        # Add references to the data for schemas
        data["references"] = [a.model_dump() for a in all_articles]
        
        # Validate with Pydantic
        report = FinalReport(**data)
        state["final_report"] = report.model_dump()
    except Exception as e:
        state["error"] = f"Summariser failed: {str(e)}"
        
    return state
