import json
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ResearchState, FinalReport
from config import get_llm


def summariser_agent(state: ResearchState) -> ResearchState:
    """
    Summarizes the top articles and builds the final report.
    """
    final_ranked_output = state.get("final_ranked_output", {})
    official_top = final_ranked_output.get("official_sources", [])
    trusted_top = final_ranked_output.get("trusted_sources", [])
    selected_urls = state.get("selected_articles", [])

    if selected_urls:
        official_top = [a for a in official_top if a.url in selected_urls]
        trusted_top = [a for a in trusted_top if a.url in selected_urls]

    original_query = state.get("original_query", "")
    all_articles = state.get("official_sources", []) + state.get(
        "trusted_sources", []
    )  # For references

    if not official_top and not trusted_top:
        return state

    prompt = ChatPromptTemplate.from_template("""
    Create a competitive intelligence research report on "{query}".

    CRITICAL RULES:
    - Generate exactly ONE insight per article provided below.
    - If a category has 3 articles, you MUST return 3 insights for it.
    - If a category has 2 articles, return 2 insights. If 1, return 1.
    - If a category is empty, return an empty array for that section.
    - Each article MUST have its own insight entry — do NOT merge articles.

    Official Sources (generate one insight per article):
    {official_articles_json}

    Trusted Sources (generate one insight per article):
    {trusted_articles_json}

    Each insight MUST:
    - Have a clear, specific title (not generic).
    - Have a brief_summary that is 5-8 sentences long.
    - Use ONLY facts from the provided article content. Do NOT hallucinate.
    - Include the exact citation_id from the article's JSON.

    You MUST return a valid JSON object with this EXACT structure:
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

    REMEMBER: Number of official_insights MUST equal number of Official Source articles.
    Number of trusted_insights MUST equal number of Trusted Source articles.
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
        official_data.append(
            {
                "citation_id": citation_id,
                "title": article.title,
                "snippet": article.snippet,
                "domain": article.domain,
            }
        )

    for article in trusted_top:
        try:
            citation_id = all_articles.index(article) + 1
        except ValueError:
            citation_id = 1
        trusted_data.append(
            {
                "citation_id": citation_id,
                "title": article.title,
                "snippet": article.snippet,
                "domain": article.domain,
            }
        )

    chain = prompt | get_llm()
    print(
        f"   [Summariser Debug] Extracted {len(official_top)} Official, {len(trusted_top)} Trusted."
    )
    response = chain.invoke(
        {
            "query": original_query,
            "official_articles_json": json.dumps(official_data),
            "trusted_articles_json": json.dumps(trusted_data),
        }
    )
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
                    if (
                        "insight" in k.lower()
                        and isinstance(v, list)
                        and k != "official_insights"
                    ):
                        trusted_insights = v
                        break
                    elif (
                        "inspection" in k.lower()
                        and isinstance(v, list)
                        and k != "official_insights"
                    ):
                        trusted_insights = v
                        break

        valid_data = {
            "report_title": report_title,
            "official_insights": official_insights,
            "trusted_insights": trusted_insights,
            "references": [a.model_dump() for a in all_articles],
        }

        # Validate with Pydantic
        report = FinalReport(**valid_data)
        state["final_report"] = report.model_dump()
    except Exception as e:
        state["error"] = f"Summariser failed: {str(e)}"
        print(f"Summariser Debug - Raw Error: {str(e)}")

    return state
