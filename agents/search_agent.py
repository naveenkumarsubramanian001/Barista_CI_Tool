import asyncio
import os
from datetime import datetime, timedelta
from typing import List
from tavily import TavilyClient
from models.schemas import ResearchState, Article
from config import TAVILY_API_KEY
from utils.date_utils import is_within_range
from utils.query_builder import build_site_query, build_trusted_query

# Initialize client at module level
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

async def perform_single_search(query: str, allowed_domains: List[str], days: int, source_type: str) -> List[dict]:
    """Helper to perform a single Tavily search asynchronously."""
    try:
        restricted_query = build_site_query(query, allowed_domains)
        print(f"🔍 Searching {source_type} for: {restricted_query} (Recency: {days} days)")
        
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
    Retrieves specifically for 'official' and 'trusted' sources independently.
    """
    subqueries = state.get("subqueries", [])
    if not subqueries:
        return state
        
    company_domains = state.get("company_domains", [])[:5]
    trusted_domains = state.get("trusted_domains", [])[:5]
    primary_entity = state.get("primary_entity", "")
    
    # Dynamic Date Range: use state override when present.
    search_days = int(state.get("search_days_used") or 180)
    
    # helper for processing a batch of results
    def process_results(search_results_list, source_type: str) -> List[Article]:
        all_articles = []
        seen_urls = set()
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
                        import tldextract
                        ext = tldextract.extract(url)
                        domain = f"{ext.domain}.{ext.suffix}" if ext.domain else None
                        
                        all_articles.append(Article(
                            title=res.get('title', 'No Title'),
                            url=url,
                            snippet=res.get('content', ''),
                            published_date=pub_date,
                            source_type=source_type,
                            domain=domain
                        ))
                        seen_urls.add(url)
                        
                if len(all_articles) >= 10:
                    break
            if len(all_articles) >= 10:
                break
                
        print(f"   - {source_type.capitalize()} search completed. Collected {len(all_articles)} unique articles.")
        return all_articles
        
    # Run official + trusted searches CONCURRENTLY for speed
    async def search_official():
        if not company_domains:
            return []
        tasks_official = [perform_single_search(q, company_domains, search_days, "official") for q in subqueries[:5]]
        search_results_official = await asyncio.gather(*tasks_official)
        return process_results(search_results_official, "official")
    
    async def search_trusted():
        if not trusted_domains:
            return []
        # Entity-anchored queries: "Company Name" site:domain1 OR site:domain2 subquery
        # This prevents broad topic-only results like 'AI startup funding rounds'
        tasks_trusted = [
            perform_single_search(
                build_trusted_query(primary_entity, q, trusted_domains),
                [],          # domains already embedded in the query by build_trusted_query
                search_days,
                "trusted"
            )
            for q in subqueries[:5]
        ]
        search_results_trusted = await asyncio.gather(*tasks_trusted)
        return process_results(search_results_trusted, "trusted")
    
    official_articles, trusted_articles = await asyncio.gather(
        search_official(), search_trusted()
    )
            
    if not official_articles and not trusted_articles:
        print("⚠️ No articles passed the filters for either official or trusted pipelines.")
            
    state["official_sources"] = official_articles
    state["trusted_sources"] = trusted_articles
    return state
