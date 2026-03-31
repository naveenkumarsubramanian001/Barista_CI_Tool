import asyncio
import json
import os
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

import tldextract

from models.schemas import ResearchState, Article
from config import SERPER_API_KEY as CONFIG_SERPER_API_KEY
from utils.date_utils import is_within_range
from utils.query_builder import build_site_query, build_trusted_query


SERPER_BASE_URL = (os.getenv("SERPER_BASE_URL") or "https://google.serper.dev").rstrip("/")


def _get_serper_api_key() -> str:
    return (CONFIG_SERPER_API_KEY or os.getenv("SERPER_API_KEY") or "").strip()


def _iso_today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _extract_domain(url: str) -> Optional[str]:
    if not url:
        return None
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return None


def _serper_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Blocking HTTP request to Serper. Call via asyncio.to_thread."""
    api_key = _get_serper_api_key()
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not set")

    url = f"{SERPER_BASE_URL}/{path.lstrip('/')}"
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def _normalize_serper_results(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize Serper responses into a Tavily-like list of dict results."""
    # Serper commonly returns one of these arrays, depending on endpoint.
    candidates = []
    for key in ("news", "organic"):
        items = raw.get(key)
        if isinstance(items, list) and items:
            candidates = items
            break

    normalized: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue

        url = item.get("link") or item.get("url")
        title = item.get("title") or "No Title"
        snippet = item.get("snippet") or item.get("description") or ""

        # Date fields vary by endpoint/result type.
        published = (
            item.get("date")
            or item.get("publishedDate")
            or item.get("published_date")
            or ""
        )

        normalized.append(
            {
                "url": url,
                "title": title,
                "content": snippet,
                "published_date": published,
            }
        )

    return normalized


async def perform_single_serper_search(
    query: str,
    allowed_domains: List[str],
    days: int,
    source_type: str,
) -> List[Dict[str, Any]]:
    """Perform a single Serper search asynchronously."""
    try:
        restricted_query = build_site_query(query, allowed_domains)
        print(f"🔍 Searching {source_type} via Serper for: {restricted_query} (Recency: {days} days)")

        # Serper doesn’t have a universal 'days' parameter; we still filter locally
        # using `is_within_range`. Using the news endpoint biases toward recency.
        payload = {
            "q": restricted_query,
            "num": 10,
        }

        # 1) Prefer /news for recency
        raw = await asyncio.to_thread(_serper_post, "/news", payload)
        results = _normalize_serper_results(raw)

        # 2) Fallback to /search if news returns nothing
        if not results:
            raw = await asyncio.to_thread(_serper_post, "/search", payload)
            results = _normalize_serper_results(raw)

        return results
    except Exception as e:
        print(f"   - Serper search failed for '{query}': {e}")
        return []


async def serper_search_agent(state: ResearchState) -> ResearchState:
    """Serper-powered version of `search_agent`.

    Mirrors the same behavior and writes:
      - state["official_sources"]: List[Article]
      - state["trusted_sources"]: List[Article]

    Optional override:
      - If state contains `search_days`, it will be used; otherwise defaults to 180.
    """

    subqueries = state.get("subqueries", [])
    if not subqueries:
        return state

    company_domains = state.get("company_domains", [])[:5]
    trusted_domains = state.get("trusted_domains", [])[:5]
    primary_entity = state.get("primary_entity", "")

    search_days = int(state.get("search_days_used") or 180)

    def process_results(search_results_list: List[List[Dict[str, Any]]], source_type: str) -> List[Article]:
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
                    # Same behavior as current Tavily agent: soft-accept missing dates.
                    # If you need strict 7-day windows, remove this branch.
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

        print(f"   - {source_type.capitalize()} Serper search completed. Collected {len(all_articles)} unique articles.")
        return all_articles

    official_articles: List[Article] = []
    if company_domains:
        tasks_official = [
            perform_single_serper_search(q, company_domains, search_days, "official")
            for q in subqueries[:5]
        ]
        search_results_official = await asyncio.gather(*tasks_official)
        official_articles = process_results(search_results_official, "official")

    trusted_articles: List[Article] = []
    if trusted_domains:
        # Entity-anchored queries anchor results to the primary company name
        tasks_trusted = [
            perform_single_serper_search(
                build_trusted_query(primary_entity, q, trusted_domains),
                [],          # domains already baked into the query string
                search_days,
                "trusted"
            )
            for q in subqueries[:5]
        ]
        search_results_trusted = await asyncio.gather(*tasks_trusted)
        trusted_articles = process_results(search_results_trusted, "trusted")

    if not official_articles and not trusted_articles:
        print("⚠️ No articles passed the filters for either official or trusted pipelines.")

    state["official_sources"] = official_articles
    state["trusted_sources"] = trusted_articles
    return state
