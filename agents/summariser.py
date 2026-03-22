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

    overview_prompt = ChatPromptTemplate.from_template("""
    Create a competitive intelligence research overview on "{query}".

    CRITICAL RULES:
    - Write an 'executive_summary' (2-3 paragraphs) synthesizing all sources to answer the main "{query}".
    - Write a 'conflict_and_consensus' section analyzing differences between official and trusted sources.

    Official Sources (Titles & Snippets only):
    {official_snippets}

    Trusted Sources (Titles & Snippets only):
    {trusted_snippets}

    You MUST return a valid JSON object with this EXACT structure:
    {{
      "report_title": "string",
      "executive_summary": "string",
      "conflict_and_consensus": "string"
    }}
    """)

    insight_prompt = ChatPromptTemplate.from_template("""
    You are an elite competitive intelligence analyst. You are evaluating a single article about "{query}".
    
    Article Domain: {domain}
    Article Title: {title}
    
    Full Text Content:
    {content}
    
    Your job is to read this entire article and extract EVERY critical fact, strategy, number, and feature into an expansive summary.
    Do not hallucinate. Do not write about other articles.
    
    You MUST return a valid JSON object with this EXACT structure:
    {{
      "title": "string (a clear, specific title for this insight)",
      "detailed_summary": "string (a highly detailed, expansive summary covering all crucial facts, numbers, and competitive positioning so the user NEVER has to read the original article)",
      "reasoning": "string (how the facts in this article answer the user's original query)",
      "sentiment": "string (e.g., 'Positive/Marketing', 'Critical/Review', 'Neutral')",
      "key_metrics": ["string"],
      "key_features": ["string"]
    }}
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

    try:
        # 1. Generate Executive Overview
        official_snippets_data = [{"title": a.title, "snippet": a.snippet} for a in official_top]
        trusted_snippets_data = [{"title": a.title, "snippet": a.snippet} for a in trusted_top]

        chain_overview = overview_prompt | get_llm()
        print("   [Summariser Debug] Generating Executive Overview...")
        res_overview = chain_overview.invoke({
            "query": original_query,
            "official_snippets": json.dumps(official_snippets_data),
            "trusted_snippets": json.dumps(trusted_snippets_data),
        })
        
        overview_data = safe_json_extract(res_overview.content)
        report_title = overview_data.get("report_title", f"Research Report: {original_query}")
        executive_summary = overview_data.get("executive_summary", "")
        conflict_and_consensus = overview_data.get("conflict_and_consensus", "")

        # 2. Process Individual Articles
        chain_insight = insight_prompt | get_llm()
        official_insights = []
        trusted_insights = []
        
        print(f"   [Summariser Debug] Processing {len(official_top)} Official Articles individually...")
        for article in official_top:
            try:
                citation_id = all_articles.index(article) + 1
            except ValueError:
                citation_id = 1
            
            full_content = get_article_content(article.url) or article.snippet
            res_insight = chain_insight.invoke({
                "query": original_query,
                "domain": article.domain,
                "title": article.title,
                "content": full_content
            })
            
            insight_data = safe_json_extract(res_insight.content)
            # Ensure safe fallback keys
            if "title" not in insight_data: insight_data["title"] = article.title
            if "detailed_summary" not in insight_data: insight_data["detailed_summary"] = "Summary missing."
            insight_data["citation_id"] = citation_id
            official_insights.append(insight_data)
            print(f"      - Processed: {article.title[:40]}...")

        print(f"   [Summariser Debug] Processing {len(trusted_top)} Trusted Articles individually...")
        for article in trusted_top:
            try:
                citation_id = all_articles.index(article) + 1
            except ValueError:
                citation_id = 1
            
            full_content = get_article_content(article.url) or article.snippet
            res_insight = chain_insight.invoke({
                "query": original_query,
                "domain": article.domain,
                "title": article.title,
                "content": full_content
            })
            
            insight_data = safe_json_extract(res_insight.content)
            if "title" not in insight_data: insight_data["title"] = article.title
            if "detailed_summary" not in insight_data: insight_data["detailed_summary"] = "Summary missing."
            insight_data["citation_id"] = citation_id
            trusted_insights.append(insight_data)
            print(f"      - Processed: {article.title[:40]}...")

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
