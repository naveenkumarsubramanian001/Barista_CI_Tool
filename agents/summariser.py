import json
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ResearchState, FinalReport
from config import get_llm


def summariser_agent(state: ResearchState) -> ResearchState:
    """
    Summarizes the top articles and builds the final report.
    """
    final_ranked_output = state.get("final_ranked_output", {})
    from models.schemas import Article
    official_top = [Article(**a) if isinstance(a, dict) else a for a in final_ranked_output.get("official_sources", [])]
    trusted_top = [Article(**a) if isinstance(a, dict) else a for a in final_ranked_output.get("trusted_sources", [])]
    selected_urls = state.get("selected_articles", [])

    if selected_urls:
        official_top = [a for a in official_top if a.url in selected_urls]
        trusted_top = [a for a in trusted_top if a.url in selected_urls]

    original_query = state.get("original_query", "")
    all_articles_tmp = state.get("official_sources", []) + state.get("trusted_sources", [])
    all_articles = [Article(**a) if isinstance(a, dict) else a for a in all_articles_tmp]  # For references

    if not official_top and not trusted_top:
        return state

    prompt = ChatPromptTemplate.from_template("""
    Create a competitive intelligence research report on "{query}".

    CRITICAL RULES:
    - First, write an 'executive_summary' (2-3 paragraphs) that synthesizes all the provided sources to answer the main "{query}".
    - Second, write a 'conflict_and_consensus' section analyzing differences between official and trusted sources.
    - Then, you MUST step through EVERY SINGLE article provided below, one by one.
    - For EVERY article in the Official Sources list, create exactly ONE object in the `official_insights` array (Total: {num_official} objects).
    - For EVERY article in the Trusted Sources list, create exactly ONE object in the `trusted_insights` array (Total: {num_trusted} objects).
    - FATAL ERROR WARNING: DO NOT skip any articles. DO NOT merge articles. Even if articles seem similar, you MUST generate a separate insight for each one to match the exact counts provided above!

    Official Sources (generate one insight per article):
    {official_articles_json}

    Trusted Sources (generate one insight per article):
    {trusted_articles_json}

    Each insight MUST:
    - Have a clear, specific title (not generic).
    - Have a detailed_summary that acts as a comprehensive brief of the article, capturing ALL the crucial facts, numbers, strategies, and competitive positioning so the user NEVER has to read the original article. It should be as detailed and expansive as possible.
    - Have a reasoning that explains how the facts in this article answer the user's original query.
    - Have a sentiment that classifies the article's stance (e.g., "Positive/Marketing", "Critical/Review", "Neutral").
    - Extract key_metrics (list of strings, e.g., prices, growth, facts) and key_features (list of strings, e.g., product features).
    - Use ONLY facts from the provided article content. Do NOT hallucinate.
    - Include the exact citation_id from the article's JSON.

    You MUST return a valid JSON object with this EXACT structure:
    {{
      "report_title": "string",
      "executive_summary": "string",
      "conflict_and_consensus": "string",
      "official_insights": [
        {{
          "title": "string",
          "detailed_summary": "string",
          "reasoning": "string",
          "sentiment": "string",
          "key_metrics": ["string"],
          "key_features": ["string"],
          "citation_id": integer
        }}
      ],
      "trusted_insights": [
        {{
          "title": "string",
          "detailed_summary": "string",
          "reasoning": "string",
          "sentiment": "string",
          "key_metrics": ["string"],
          "key_features": ["string"],
          "citation_id": integer
        }}
      ]
    }}

    REMEMBER: Number of official_insights MUST equal number of Official Source articles.
    Number of trusted_insights MUST equal number of Trusted Source articles.
    """)

    from utils.json_utils import safe_json_extract
    import requests
    from bs4 import BeautifulSoup

    def get_article_content(url):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            res = requests.get(url, headers=headers, timeout=5)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()
            article_tag = soup.find("article") or soup.find("main")
            if article_tag:
                text_content = article_tag.get_text(separator="\n", strip=True)
            else:
                paragraphs = soup.find_all("p")
                text_content = "\n\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 60)
            return text_content[:4000] if text_content else None
        except Exception:
            return None

    official_data = []
    trusted_data = []

    for article in official_top:
        # Safe fallback index if article is somehow not in the combined list
        try:
            citation_id = all_articles.index(article) + 1
        except ValueError:
            citation_id = 1
            
        full_content = get_article_content(article.url) or article.snippet
        official_data.append(
            {
                "citation_id": citation_id,
                "title": article.title,
                "content": full_content,
                "domain": article.domain,
            }
        )

    for article in trusted_top:
        try:
            citation_id = all_articles.index(article) + 1
        except ValueError:
            citation_id = 1
            
        full_content = get_article_content(article.url) or article.snippet
        trusted_data.append(
            {
                "citation_id": citation_id,
                "title": article.title,
                "content": full_content,
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
            "num_official": len(official_top),
            "num_trusted": len(trusted_top),
        }
    )
    print(f"   [Summariser Debug] LLM Raw Response:\n{response.content}")

    try:
        # Use safe_json_extract for robustness
        data = safe_json_extract(response.content)

        # Robust parsing for Qwen LLM hallucinations
        report_title = data.get("report_title", f"Research Report: {original_query}")
        executive_summary = data.get("executive_summary", "")
        conflict_and_consensus = data.get("conflict_and_consensus", "")

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
            "executive_summary": executive_summary,
            "conflict_and_consensus": conflict_and_consensus,
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
