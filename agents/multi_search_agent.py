"""
Multi-Source Search Aggregator with Rich logging.

Runs ALL available search APIs in parallel, merges results,
deduplicates using URL matching + embedding similarity, and
produces a single combined article pool for the discriminator.
"""

import asyncio
from typing import Any, List, Optional, Tuple

from models.schemas import ResearchState, Article
from config import (
    TAVILY_API_KEY,
    SERPER_API_KEY,
    BING_SEARCH_API_KEY,
    GOOGLE_API_KEY,
    GOOGLE_CSE_ID,
    SEARCH_PROVIDER,
    SEARCH_STRATEGY,
)
from utils.logger import (
    console, banner, section, info, success, warning, error,
    detail, provider_table, article_table, merge_summary, phase_progress
)


def _get_available_providers() -> List[Tuple[str, callable]]:
    """Discover which search providers have API keys configured."""
    providers = []

    if TAVILY_API_KEY:
        from agents.search_agent import search_agent
        providers.append(("Tavily", search_agent))

    if SERPER_API_KEY:
        from agents.serper_search_agent import serper_search_agent
        providers.append(("Serper", serper_search_agent))

    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        from agents.google_search_agent import google_search_agent
        providers.append(("Google", google_search_agent))

    if BING_SEARCH_API_KEY:
        from agents.bing_search_agent import bing_search_agent
        providers.append(("Bing", bing_search_agent))

    return providers


def _provider_order(providers: List[Tuple[str, callable]]) -> List[Tuple[str, callable]]:
    """Return provider order honoring configured primary provider."""
    if not providers:
        return []

    name_map = {name.lower(): (name, fn) for name, fn in providers}
    ordered: List[Tuple[str, callable]] = []

    primary = name_map.get(SEARCH_PROVIDER)
    if primary:
        ordered.append(primary)

    for name, fn in providers:
        pair = (name, fn)
        if pair not in ordered:
            ordered.append(pair)

    return ordered


def _as_article(item: Any, source_type: Optional[str] = None) -> Optional[Article]:
    """Normalize provider output item to Article."""
    if isinstance(item, Article):
        if source_type and not item.source_type:
            item.source_type = source_type
        return item
    if isinstance(item, dict):
        payload = dict(item)
        if source_type and not payload.get("source_type"):
            payload["source_type"] = source_type
        try:
            return Article(**payload)
        except Exception:
            return None
    return None


def _dedup_by_url(articles: List[Article]) -> List[Article]:
    """Remove exact URL duplicates, keeping the first occurrence."""
    seen = set()
    unique = []
    for article in articles:
        normalized = article.url.rstrip("/").lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(article)
    return unique


def _dedup_by_similarity(articles: List[Article], threshold: float = 0.92) -> List[Article]:
    """Remove near-duplicate articles using embedding similarity."""
    if len(articles) <= 1:
        return articles

    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        from config import get_embedding_model

        embed_model = get_embedding_model()
        texts = [f"{a.title} {a.snippet[:200]}" for a in articles]
        embeddings = np.array(embed_model.embed_documents(texts))
        sim_matrix = cosine_similarity(embeddings)

        to_remove = set()
        for i in range(len(articles)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(articles)):
                if j in to_remove:
                    continue
                if sim_matrix[i][j] > threshold:
                    to_remove.add(j)
                    detail(f"Dedup: '{articles[j].title[:40]}' ≈ '{articles[i].title[:40]}' (sim={sim_matrix[i][j]:.3f})")

        deduped = [a for idx, a in enumerate(articles) if idx not in to_remove]
        info(f"Similarity dedup: {len(articles)} → {len(deduped)} (removed {len(to_remove)} near-duplicates)")
        return deduped

    except Exception as e:
        warning(f"Similarity dedup skipped: {e}")
        return articles


async def multi_search_agent(state: ResearchState) -> ResearchState:
    """
    Multi-Source Search Aggregator.
    Runs ALL available search providers in parallel, merges, and deduplicates.
    """
    providers = _provider_order(_get_available_providers())

    if not providers:
        error("No search API keys configured! Cannot search.")
        state["error"] = "No search API keys available."
        state.setdefault("logs", []).append("❌ No search providers configured.")
        return state

    provider_names = [name for name, _ in providers]
    
    section("Multi-Source Search", "🔎")
    provider_table(provider_names)
    state.setdefault("logs", []).append(
        f"🔧 Search strategy: {SEARCH_STRATEGY} (primary: {SEARCH_PROVIDER}, available: {', '.join(provider_names)})"
    )

    async def run_provider(name: str, agent_func) -> Tuple[str, List[Article], List[Article]]:
        try:
            provider_state = dict(state)
            provider_state["official_sources"] = []
            provider_state["trusted_sources"] = []
            result_state = await agent_func(provider_state)
            official_raw = result_state.get("official_sources", [])
            trusted_raw = result_state.get("trusted_sources", [])

            official = [a for a in (_as_article(x, "official") for x in official_raw) if a]
            trusted = [a for a in (_as_article(x, "trusted") for x in trusted_raw) if a]

            success(f"{name}: {len(official)} official + {len(trusted)} trusted")
            return (name, official, trusted)
        except Exception as e:
            error(f"{name} failed: {e}")
            return (name, [], [])

    results: List[Tuple[str, List[Article], List[Article]]] = []
    strategy = SEARCH_STRATEGY if SEARCH_STRATEGY in {"parallel", "single", "fallback"} else "parallel"

    if strategy == "single":
        name, func = providers[0]
        with phase_progress(f"Running single provider: {name}"):
            results = [await run_provider(name, func)]
    elif strategy == "fallback":
        with phase_progress("Running providers in fallback mode"):
            for name, func in providers:
                current = await run_provider(name, func)
                results.append(current)
                _, off, tr = current
                if off or tr:
                    state["logs"].append(f"✅ Fallback succeeded with provider: {name}")
                    break
    else:
        with phase_progress("Running all search providers in parallel"):
            results = await asyncio.gather(*[run_provider(name, func) for name, func in providers])

    all_official = []
    all_trusted = []
    for name, official, trusted in results:
        all_official.extend(official)
        all_trusted.extend(trusted)

    raw_total = len(all_official) + len(all_trusted)

    # Stage 1: URL dedup
    all_official = _dedup_by_url(all_official)
    all_trusted = _dedup_by_url(all_trusted)
    after_url = len(all_official) + len(all_trusted)

    # Stage 2: Similarity dedup
    if len(all_official) > 1:
        all_official = _dedup_by_similarity(all_official)
    if len(all_trusted) > 1:
        all_trusted = _dedup_by_similarity(all_trusted)

    final_total = len(all_official) + len(all_trusted)
    merge_summary(raw_total, after_url, final_total)

    # Show combined article table
    all_combined = all_official + all_trusted
    if all_combined:
        article_table(all_combined, "📰 Combined Article Pool")

    state["official_sources"] = [a.model_dump() for a in all_official]
    state["trusted_sources"] = [a.model_dump() for a in all_trusted]

    if not all_official and not all_trusted:
        state.setdefault("logs", []).append("⚠️ Providers ran, but no results survived dedup/filtering.")

    return state
