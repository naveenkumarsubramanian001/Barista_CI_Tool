import asyncio
import re
from datetime import datetime
from typing import List
from tavily import TavilyClient
from models.schemas import ResearchState, Article
from config import TAVILY_API_KEY
from utils.date_utils import is_within_range
from utils.query_builder import build_site_query

# Initialize client at module level
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


async def perform_single_search(
    query: str, allowed_domains: List[str], days: int, source_type: str
) -> List[dict]:
    """Helper to perform a single Tavily search asynchronously."""
    try:
        # ALWAYS restrict query to the allowed domains to ensure precise targeting for both official and trusted
        restricted_query = build_site_query(query, allowed_domains)

        print(
            f"🔍 Searching {source_type} for: {restricted_query} (Recency: {days} days)"
        )

        kwargs = {
            "query": restricted_query,
            "search_depth": "advanced",
            "max_results": 5,
        }

        if source_type == "trusted":
            kwargs["topic"] = "news"
            kwargs["days"] = days
            
            response = await asyncio.to_thread(tavily_client.search, **kwargs)
            results = response.get("results", [])
            
            # Fallback to general web search if 'news' topic yields nothing (common for B2B)
            if not results:
                print(f"   ⚠️ No 'news' found for '{query}'. Falling back to 'general' web search...")
                kwargs["topic"] = "general"
                kwargs.pop("days", None) # general search does not strictly enforce 'days'
                response = await asyncio.to_thread(tavily_client.search, **kwargs)
                results = response.get("results", [])
                
            return results
        else:
            kwargs["topic"] = "general"
            response = await asyncio.to_thread(tavily_client.search, **kwargs)
            return response.get("results", [])

    except Exception as e:
        print(f"   - Search failed for '{query}': {e}")
        return []


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
    pattern = r"(?:last|past)\s+(\d+)\s*(day|week|month|year)s?"
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
    if re.search(r"(?:last|past)\s+year", q):
        return 365

    # "last month" / "past month" without a number
    if re.search(r"(?:last|past)\s+month", q):
        return 30

    # "last week" / "past week" without a number
    if re.search(r"(?:last|past)\s+week", q):
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
    state.setdefault("logs", [])

    # helper for processing a batch of results
    def process_results(search_results_list, source_type: str) -> List[Article]:
        all_articles = []
        seen_urls = set()
        max_results_per_subquery = 5

        for i in range(max_results_per_subquery):
            for results in search_results_list:
                if i < len(results):
                    res = results[i]
                    url = res.get("url")

                    if url in seen_urls:
                        continue

                    pub_date = res.get("published_date", "")

                    # Validation Logic
                    valid = False
                    if is_within_range(pub_date, search_days):
                        valid = True
                    elif not pub_date:
                        # If news topic was used, we soft-accept missing dates tagged as today
                        valid = True
                        pub_date = datetime.now().strftime("%Y-%m-%d")

                    if valid:
                        import tldextract

                        ext = tldextract.extract(url)
                        domain = f"{ext.domain}.{ext.suffix}" if ext.domain else None

                        all_articles.append(
                            Article(
                                title=res.get("title", "No Title"),
                                url=url,
                                snippet=res.get("content", ""),
                                published_date=pub_date,
                                source_type=source_type,
                                domain=domain,
                            )
                        )
                        seen_urls.add(url)

                if len(all_articles) >= 10:
                    break
            if len(all_articles) >= 10:
                break

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

    state["logs"].append(f"🔎 Starting deep search for {len(subqueries)} topics...")
    state["logs"].append(f"📅 Primary search window: {search_days} days.")

    if company_domains or trusted_domains:
        state["logs"].append(
            f"🎯 Targeting {len(company_domains)} official and {len(trusted_domains)} trusted domains."
        )

    for attempt_days in time_windows:
        if attempt_days != time_windows[0]:
            state["logs"].append(
                f"🔄 Widening search window to {attempt_days} days (fallback)..."
            )

        # 1. Execute parallel searches for Official Domains
        if company_domains and not official_articles:
            state["logs"].append(
                f"🏢 Searching official company sources ({attempt_days}d)..."
            )
            tasks_official = [
                perform_single_search(q, company_domains, attempt_days, "official")
                for q in subqueries[:5]
            ]
            search_results_official = await asyncio.gather(*tasks_official)
            official_articles = process_results(search_results_official, "official")

        # 2. Execute parallel searches for Trusted Domains
        if trusted_domains and not trusted_articles:
            state["logs"].append(
                f"📰 Searching trusted news and analysis ({attempt_days}d)..."
            )
            tasks_trusted = [
                perform_single_search(q, trusted_domains, attempt_days, "trusted")
                for q in subqueries[:5]
            ]
            search_results_trusted = await asyncio.gather(*tasks_trusted)
            trusted_articles = process_results(search_results_trusted, "trusted")

        if official_articles or trusted_articles:
            state["logs"].append(
                f"✨ Found {len(official_articles) + len(trusted_articles)} relevant articles."
            )
            break  # Got results, stop widening

    if not official_articles and not trusted_articles:
        state["logs"].append(
            "⚠️ No articles found even after widening search timeframe."
        )

    state["official_sources"] = [a.model_dump() for a in official_articles]
    state["trusted_sources"] = [a.model_dump() for a in trusted_articles]
    state["search_days_used"] = attempt_days  # Track which window worked
    return state
