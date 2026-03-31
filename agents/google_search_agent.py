"""
Google Custom Search JSON API agent.
Uses the Programmable Search Engine (formerly Custom Search Engine).

Setup:
  1. Enable "Custom Search API" in Google Cloud Console
  2. Create API key at console.cloud.google.com/apis/credentials
  3. Create search engine at programmablesearchengine.google.com
  4. Set GOOGLE_API_KEY and GOOGLE_CSE_ID in .env

Free tier: 100 queries/day. Paid: $5 per 1000 queries.
"""

import asyncio
import json
import os
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Any, Dict, List, Optional

import tldextract

from models.schemas import ResearchState, Article
from utils.date_utils import is_within_range
from utils.query_builder import build_site_query, build_trusted_query

GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
GOOGLE_CSE_ID = (os.getenv("GOOGLE_CSE_ID") or "").strip()
GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"


def _iso_today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _extract_domain(url: str) -> Optional[str]:
    if not url:
        return None
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return None


def _google_get(params: Dict[str, Any]) -> Dict[str, Any]:
    """Blocking HTTP GET to Google Custom Search API. Call via asyncio.to_thread."""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise RuntimeError("GOOGLE_API_KEY or GOOGLE_CSE_ID is not set")

    params["key"] = GOOGLE_API_KEY
    params["cx"] = GOOGLE_CSE_ID

    query_string = urllib.parse.urlencode(params)
    url = f"{GOOGLE_CSE_URL}?{query_string}"

    req = urllib.request.Request(
        url=url,
        method="GET",
        headers={"Accept": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _normalize_google_results(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Google Custom Search API response to standard format."""
    results = []
    for item in raw.get("items", []):
        url = item.get("link", "")
        title = item.get("title", "No Title")
        snippet = item.get("snippet", "")

        # Extract date from pagemap or metatags if available
        published = ""
        pagemap = item.get("pagemap", {})
        metatags = pagemap.get("metatags", [{}])
        if metatags and isinstance(metatags, list):
            meta = metatags[0] if metatags else {}
            published = (
                meta.get("article:published_time", "")
                or meta.get("og:updated_time", "")
                or meta.get("date", "")
                or ""
            )
        if published and "T" in published:
            published = published.split("T")[0]

        results.append({
            "url": url,
            "title": title,
            "content": snippet,
            "published_date": published,
        })

    return results


async def perform_single_google_search(
    query: str,
    allowed_domains: List[str],
    days: int,
    source_type: str,
) -> List[Dict[str, Any]]:
    """Perform a single Google Custom Search asynchronously."""
    try:
        restricted_query = build_site_query(query, allowed_domains)
        print(f"🔍 Searching {source_type} via Google for: {restricted_query} (Recency: {days} days)")

        # Google CSE params
        params = {
            "q": restricted_query,
            "num": 10,
            "sort": "date",  # Sort by date for recency
        }

        # Recency filter: dateRestrict parameter
        # d[number] = days, w[number] = weeks, m[number] = months
        if days <= 7:
            params["dateRestrict"] = f"d{days}"
        elif days <= 30:
            params["dateRestrict"] = f"w{days // 7}"
        else:
            params["dateRestrict"] = f"m{days // 30}"

        raw = await asyncio.to_thread(_google_get, params)
        results = _normalize_google_results(raw)

        return results
    except Exception as e:
        print(f"   - Google search failed for '{query}': {e}")
        return []


async def google_search_agent(state: ResearchState) -> ResearchState:
    """
    Google Custom Search powered agent.
    Same interface as search_agent, serper_search_agent, bing_search_agent.
    Writes to state["official_sources"] and state["trusted_sources"].
    """
    subqueries = state.get("subqueries", [])
    if not subqueries:
        return state

    company_domains = state.get("company_domains", [])[:5]
    trusted_domains = state.get("trusted_domains", [])[:5]
    primary_entity = state.get("primary_entity", "")

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

        print(f"   - {source_type.capitalize()} Google search completed. Collected {len(all_articles)} unique articles.")
        return all_articles

    # Run official + trusted searches CONCURRENTLY
    async def search_official():
        if not company_domains:
            return []
        tasks = [
            perform_single_google_search(q, company_domains, search_days, "official")
            for q in subqueries[:5]
        ]
        results = await asyncio.gather(*tasks)
        return process_results(results, "official")

    async def search_trusted():
        if not trusted_domains:
            return []
        # Entity-anchored queries anchor results to the primary company name
        tasks = [
            perform_single_google_search(
                build_trusted_query(primary_entity, q, trusted_domains),
                [],          # domains already baked into the query string
                search_days,
                "trusted"
            )
            for q in subqueries[:5]
        ]
        results = await asyncio.gather(*tasks)
        return process_results(results, "trusted")

    official_articles, trusted_articles = await asyncio.gather(
        search_official(), search_trusted()
    )

    if not official_articles and not trusted_articles:
        print("⚠️ No articles passed the filters for either official or trusted pipelines (Google).")

    state["official_sources"] = official_articles
    state["trusted_sources"] = trusted_articles
    return state
