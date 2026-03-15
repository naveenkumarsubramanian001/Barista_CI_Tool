import json
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ResearchState, FinalReport, Insight
from config import get_llm

def summariser_agent(state: ResearchState) -> ResearchState:
    """
    Summarizes the top articles and builds the final report.
    """
    final_ranked_output = state.get("final_ranked_output", {})
    official_top = final_ranked_output.get("official_sources", [])
    trusted_top = final_ranked_output.get("trusted_sources", [])
    
    original_query = state.get("original_query", "")
    all_articles = state.get("official_sources", []) + state.get("trusted_sources", []) # For references
    
    if not official_top and not trusted_top:
        return state
        
    prompt = ChatPromptTemplate.from_template("""
    Create a research report on "{query}".
    
    You must generate insights from the provided articles grouped by their source type. 
    If a source type array is empty, return an empty array for that section.
    If it has articles, YOU MUST RETURN AT LEAST ONE INSIGHT for that section!
    
    Official Sources:
    {official_articles_json}
    
    Trusted Sources:
    {trusted_articles_json}
    
    Each insight summary must:
    - Be 5-8 sentences long.
    - Use ONLY provided content.
    - Include a citation_id mapping to the "citation_id" provided in the JSON.
    
    You MUST return a valid JSON object with the following structure:
    {{
      "report_title": "string",
      "official_insights": [
        {{
          "title": "string", 
          "brief_summary": "string", 
          "citation_id": integer
        }}
      ],
      "trusted_insights": [
        {{
          "title": "string", 
          "brief_summary": "string", 
          "citation_id": integer
        }}
      ]
    }}
    """)
    
    from utils.json_utils import safe_json_extract
    
    official_data = []
    trusted_data = []
    
    for article in official_top:
        # Safe fallback index if article is somehow not in the combined list
        try:
            citation_id = all_articles.index(article) + 1
        except ValueError:
            citation_id = 1
        official_data.append({
            "citation_id": citation_id,
            "title": article.title,
            "snippet": article.snippet,
            "domain": article.domain
        })
        
    for article in trusted_top:
        try:
            citation_id = all_articles.index(article) + 1
        except ValueError:
            citation_id = 1
        trusted_data.append({
            "citation_id": citation_id,
            "title": article.title,
            "snippet": article.snippet,
            "domain": article.domain
        })
        
    chain = prompt | get_llm()
    print(f"   [Summariser Debug] Extracted {len(official_top)} Official, {len(trusted_top)} Trusted.")
    response = chain.invoke({
        "query": original_query,
        "official_articles_json": json.dumps(official_data),
        "trusted_articles_json": json.dumps(trusted_data)
    })
    print(f"   [Summariser Debug] LLM Raw Response:\n{response.content}")
    
    try:
        # Use safe_json_extract for robustness
        data = safe_json_extract(response.content)
        
        # Robust parsing for Qwen LLM hallucinations
        report_title = data.get("report_title", f"Research Report: {original_query}")
        
        official_insights = data.get("official_insights", [])
        trusted_insights = data.get("trusted_insights", [])
        
        # Fallback heuristic matching if keys were generated incorrectly
        if not official_insights and official_top:
            for k, v in data.items():
                if "official" in k.lower() and isinstance(v, list):
                    official_insights = v
                    break
                    
        if not trusted_insights and trusted_top:
            for k, v in data.items():
                if "trusted" in k.lower() and isinstance(v, list):
                    trusted_insights = v
                    break
            # Extreme fallback: assign lists containing 'insight' or 'inspection'
            if not trusted_insights:
                for k, v in data.items():
                    if "insight" in k.lower() and isinstance(v, list) and k != "official_insights":
                        trusted_insights = v
                        break
                    elif "inspection" in k.lower() and isinstance(v, list) and k != "official_insights":
                        trusted_insights = v
                        break
                        
        valid_data = {
            "report_title": report_title,
            "official_insights": official_insights,
            "trusted_insights": trusted_insights,
            "references": [a.model_dump() for a in all_articles]
        }
            
        # Validate with Pydantic
        report = FinalReport(**valid_data)
        state["final_report"] = report.model_dump()
    except Exception as e:
        state["error"] = f"Summariser failed: {str(e)}"
        print(f"Summariser Debug - Raw Error: {str(e)}")
        
    return state
