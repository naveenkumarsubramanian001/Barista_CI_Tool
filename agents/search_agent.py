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

import re

def _extract_days_from_query(query: str) -> int:
    """
    Parse the user query for an explicit timeframe.
    Returns the number of days if found, otherwise the default of 30.

    Supported patterns:
      - "last N days/weeks/months/years"
      - "past N days/weeks/months/years"
      - "N day/week/month/year"
      - "recent" (without a number) → default 30
    """
    default_days = 30
    q = query.lower()

    # Match patterns like "last 6 months", "past 2 weeks", "last year"
    pattern = r'(?:last|past)\s+(\d+)\s*(day|week|month|year)s?'
    match = re.search(pattern, q)
    if match:
        n = int(match.group(1))
        unit = match.group(2)
        if unit == "day":
            return n
        elif unit == "week":
            return n * 7
        elif unit == "month":
            return n * 30
        elif unit == "year":
            return n * 365

    # "last year" / "past year" without a number
    if re.search(r'(?:last|past)\s+year', q):
        return 365

    # "last month" / "past month" without a number
    if re.search(r'(?:last|past)\s+month', q):
        return 30

    # "last week" / "past week" without a number
    if re.search(r'(?:last|past)\s+week', q):
        return 7

    return default_days


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
    
    # Dynamic Date Range: default 30 days, overridden only if query mentions a timeframe
    original_query = state.get("original_query", "")
    search_days = _extract_days_from_query(original_query)
    print(f"📅 Search recency: {search_days} days (from query: \"{original_query}\")")
    
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
        
    # ── Progressive time-window widening ──────────────────────────────
    # Build the list of windows to try. Start with the query-derived default,
    # then widen to 90d and 180d only if no results are found.
    time_windows = [search_days]
    for fallback in [90, 180]:
        if fallback > search_days and fallback not in time_windows:
            time_windows.append(fallback)

    official_articles = []
    trusted_articles = []

    for attempt_days in time_windows:
        if attempt_days != time_windows[0]:
            print(f"\n🔄 Widening search window to {attempt_days} days (fallback)...")

        # 1. Execute parallel searches for Official Domains
        if company_domains and not official_articles:
            tasks_official = [perform_single_search(q, company_domains, attempt_days, "official") for q in subqueries[:5]]
            search_results_official = await asyncio.gather(*tasks_official)
            official_articles = process_results(search_results_official, "official")

        # 2. Execute parallel searches for Trusted Domains
        if trusted_domains and not trusted_articles:
            tasks_trusted = [perform_single_search(q, trusted_domains, attempt_days, "trusted") for q in subqueries[:5]]
            search_results_trusted = await asyncio.gather(*tasks_trusted)
            trusted_articles = process_results(search_results_trusted, "trusted")

        if official_articles or trusted_articles:
            print(f"✅ Found articles at {attempt_days}-day window.")
            break  # Got results, stop widening

    if not official_articles and not trusted_articles:
        print("⚠️ No articles found even after widening to 180 days.")

    state["official_sources"] = official_articles
    state["trusted_sources"] = trusted_articles
    state["search_days_used"] = attempt_days  # Track which window worked
    return state
