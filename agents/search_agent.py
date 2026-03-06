import asyncio
import os
from datetime import datetime, timedelta
from typing import List
from tavily import TavilyClient
from models.schemas import ResearchState, Article
from config import TAVILY_API_KEY
from utils.date_utils import is_within_range
from utils.query_builder import build_site_query

# Initialize client at module level
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

async def perform_single_search(query: str, allowed_domains: List[str], days: int) -> List[dict]:
    """Helper to perform a single Tavily search asynchronously."""
    try:
        restricted_query = build_site_query(query, allowed_domains)
        print(f"🔍 Searching for: {restricted_query} (Recency: {days} days)")
        
        # topic='news' is excellent for recent advancements
        response = await asyncio.to_thread(
            tavily_client.search,
            query=restricted_query,
            search_depth="advanced",
            topic="news",
            days=days,
            max_results=5
        )
        return response.get('results', [])
    except Exception as e:
        print(f"   - Search failed for '{query}': {e}")
        return []

async def search_agent(state: ResearchState) -> ResearchState:
    """
    Executes parallel searches and filters results based on dynamic timeframe.
    """
    subqueries = state.get("subqueries", [])
    if not subqueries:
        return state
        
    # Combine domains
    company_domains = state.get("company_domains", [])
    trusted_domains = state.get("trusted_domains", [])
    allowed_domains = list(set(company_domains + trusted_domains))[:10]
    
    # Dynamic Date Range: Default to 180 days (6 months) as per user goal
    # In a more advanced version, we could extract this from the query analysis
    search_days = 180 
    
    # 1. Execute parallel searches
    tasks = [perform_single_search(q, allowed_domains, search_days) for q in subqueries[:5]]
    search_results_list = await asyncio.gather(*tasks)
    
    # 2. Process and filter results
    all_articles = []
    seen_urls = set()
    
    # Flatten and prioritize diversity (one from each task first, then second, etc.)
    max_results_per_subquery = 5
    for i in range(max_results_per_subquery):
        for results in search_results_list:
            if i < len(results):
                res = results[i]
                url = res.get('url')
                
                if url in seen_urls:
                    continue
                
                pub_date = res.get('published_date', '')
                
                # Validation Logic
                valid = False
                if is_within_range(pub_date, search_days):
                    valid = True
                elif not pub_date:
                    # If news topic was used, we soft-accept missing dates tagged as today
                    valid = True
                    pub_date = datetime.now().strftime('%Y-%m-%d')
                
                if valid:
                    all_articles.append(Article(
                        title=res.get('title', 'No Title'),
                        url=url,
                        snippet=res.get('content', ''),
                        published_date=pub_date
                    ))
                    seen_urls.add(url)
                    
            if len(all_articles) >= 10:
                break
        if len(all_articles) >= 10:
            break
            
    print(f"   - Search completed. Collected {len(all_articles)} unique articles.")
    if not all_articles:
        print("⚠️ No articles passed the filters.")
            
    state["articles"] = all_articles
    return state
