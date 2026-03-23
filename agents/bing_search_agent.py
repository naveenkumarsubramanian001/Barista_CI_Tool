"""
Bing Web Search API v7 search agent.
Same interface as search_agent.py and serper_search_agent.py.
Falls back gracefully if no BING_SEARCH_API_KEY is set.
"""

import asyncio
import json
import os
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

import tldextract

from models.schemas import ResearchState, Article
from utils.date_utils import is_within_range
from utils.query_builder import build_site_query

BING_API_KEY = (os.getenv("BING_SEARCH_API_KEY") or "").strip()
BING_NEWS_URL = "https://api.bing.microsoft.com/v7.0/news/search"
BING_WEB_URL = "https://api.bing.microsoft.com/v7.0/search"


def _iso_today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _extract_domain(url: str) -> Optional[str]:
    if not url:
        return None
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return None


def _bing_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Blocking HTTP GET to Bing API. Call via asyncio.to_thread."""
    if not BING_API_KEY:
        raise RuntimeError("BING_SEARCH_API_KEY is not set")

    query_string = urllib.parse.urlencode(params)
    url = f"{endpoint}?{query_string}"

    req = urllib.request.Request(
        url=url,
        method="GET",
        headers={
            "Ocp-Apim-Subscription-Key": BING_API_KEY,
            "Accept": "application/json",
        },
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _normalize_bing_news(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Bing News API response."""
    results = []
    for item in raw.get("value", []):
        url = item.get("url", "")
        title = item.get("name", "No Title")
        snippet = item.get("description", "")
        published = item.get("datePublished", "")
        # Bing datePublished is ISO 8601, extract date part
        if published and "T" in published:
            published = published.split("T")[0]
        results.append({
            "url": url,
            "title": title,
            "content": snippet,
            "published_date": published,
        })
    return results


def _normalize_bing_web(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Bing Web Search API response."""
    results = []
    web_pages = raw.get("webPages", {}).get("value", [])
    for item in web_pages:
        url = item.get("url", "")
        title = item.get("name", "No Title")
        snippet = item.get("snippet", "")
        published = item.get("dateLastCrawled", "")
        if published and "T" in published:
            published = published.split("T")[0]
        results.append({
            "url": url,
            "title": title,
            "content": snippet,
            "published_date": published,
        })
    return results


async def perform_single_bing_search(
    query: str,
    allowed_domains: List[str],
    days: int,
    source_type: str,
) -> List[Dict[str, Any]]:
    """Perform a single Bing search asynchronously."""
    try:
        restricted_query = build_site_query(query, allowed_domains)
        print(f"🔍 Searching {source_type} via Bing for: {restricted_query} (Recency: {days} days)")

        params = {
            "q": restricted_query,
            "count": 10,
            "mkt": "en-US",
            "freshness": "Month",  # Bing freshness: Day, Week, Month
        }

        # 1) Try news endpoint first for recency
        try:
            raw = await asyncio.to_thread(_bing_get, BING_NEWS_URL, params)
            results = _normalize_bing_news(raw)
        except Exception:
            results = []

        # 2) Fallback to web search if news is empty
        if not results:
            raw = await asyncio.to_thread(_bing_get, BING_WEB_URL, params)
            results = _normalize_bing_web(raw)

        return results
    except Exception as e:
        print(f"   - Bing search failed for '{query}': {e}")
        return []


async def bing_search_agent(state: ResearchState) -> ResearchState:
    """
    Bing-powered search agent. Same interface as search_agent and serper_search_agent.
    Writes to state["official_sources"] and state["trusted_sources"].
    """
    subqueries = state.get("subqueries", [])
    if not subqueries:
        return state

    company_domains = state.get("company_domains", [])[:5]
    trusted_domains = state.get("trusted_domains", [])[:5]

    search_days = int(state.get("search_days_used") or 180)

    def process_results(
        search_results_list: List[List[Dict[str, Any]]], source_type: str
    ) -> List[Article]:
        all_articles: List[Article] = []
        seen_urls = set()
        max_results_per_subquery = 5

        for i in range(max_results_per_subquery):
            for results in search_results_list:
                if i >= len(results):
                    continue

                res = results[i]
                url = res.get("url")
                if not url or url in seen_urls:
                    continue

                pub_date = res.get("published_date", "") or ""

                valid = False
                if is_within_range(pub_date, search_days):
                    valid = True
                elif not pub_date:
                    valid = True
                    pub_date = _iso_today()

                if not valid:
                    continue

                all_articles.append(
                    Article(
                        title=res.get("title", "No Title"),
                        url=url,
                        snippet=res.get("content", ""),
                        published_date=pub_date,
                        source_type=source_type,
                        domain=_extract_domain(url),
                    )
                )
                seen_urls.add(url)

                if len(all_articles) >= 10:
                    break
            if len(all_articles) >= 10:
                break

        print(f"   - {source_type.capitalize()} Bing search completed. Collected {len(all_articles)} unique articles.")
        return all_articles

    # Run official + trusted searches CONCURRENTLY
    async def search_official():
        if not company_domains:
            return []
        tasks = [
            perform_single_bing_search(q, company_domains, search_days, "official")
            for q in subqueries[:5]
        ]
        results = await asyncio.gather(*tasks)
        return process_results(results, "official")

    async def search_trusted():
        if not trusted_domains:
            return []
        tasks = [
            perform_single_bing_search(q, trusted_domains, search_days, "trusted")
            for q in subqueries[:5]
        ]
        results = await asyncio.gather(*tasks)
        return process_results(results, "trusted")

    official_articles, trusted_articles = await asyncio.gather(
        search_official(), search_trusted()
    )

    if not official_articles and not trusted_articles:
        print("⚠️ No articles passed the filters for either official or trusted pipelines (Bing).")

    state["official_sources"] = official_articles
    state["trusted_sources"] = trusted_articles
    return state
