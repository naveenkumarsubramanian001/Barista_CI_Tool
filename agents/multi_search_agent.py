"""
Multi-Source Search Aggregator with Rich logging.

Runs ALL available search APIs in parallel, merges results,
deduplicates using URL matching + embedding similarity, and
produces a single combined article pool for the discriminator.
"""

import asyncio
import os
from typing import Dict, List, Tuple

from models.schemas import ResearchState, Article
from config import TAVILY_API_KEY, SERPER_API_KEY, BING_SEARCH_API_KEY, GOOGLE_API_KEY
from utils.logger import (
    console, banner, section, info, success, warning, error,
    detail, provider_table, article_table, merge_summary, phase_progress
)
from utils.query_builder import is_entity_relevant


def _get_available_providers() -> List[Tuple[str, callable]]:
    """Discover which search providers have API keys configured."""
    providers = []

    if TAVILY_API_KEY:
        from agents.search_agent import search_agent
        providers.append(("Tavily", search_agent))

    if SERPER_API_KEY:
        from agents.serper_search_agent import serper_search_agent
        providers.append(("Serper", serper_search_agent))

    if GOOGLE_API_KEY:
        from agents.google_search_agent import google_search_agent
        providers.append(("Google", google_search_agent))

    if BING_SEARCH_API_KEY:
        from agents.bing_search_agent import bing_search_agent
        providers.append(("Bing", bing_search_agent))

    return providers


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


def _candidate_search_windows(initial_days: int | None) -> List[int]:
    if initial_days is None:
        windows = [15, 30, 60, 90, 180]
    else:
        start = int(initial_days)
        if start < 15:
            windows = [start, 15, 30, 60, 90, 180]
        elif start == 15:
            windows = [15, 30, 60, 90, 180]
        else:
            windows = [start, 30, 60, 90, 180]

    ordered: List[int] = []
    for days in windows:
        if days not in ordered:
            ordered.append(days)
    return ordered


async def _run_provider_pass(state: ResearchState, search_days: int) -> ResearchState:
    providers = _get_available_providers()

    provider_names = [name for name, _ in providers]
    section("Multi-Source Search", "🔎")
    provider_table(provider_names)

    async def run_provider(name: str, agent_func) -> Tuple[str, List[Article], List[Article]]:
        try:
            provider_state = dict(state)
            provider_state["official_sources"] = []
            provider_state["trusted_sources"] = []
            provider_state["search_days_used"] = search_days
            result_state = await agent_func(provider_state)
            official = result_state.get("official_sources", [])
            trusted = result_state.get("trusted_sources", [])
            success(f"{name}: {len(official)} official + {len(trusted)} trusted")
            return (name, official, trusted)
        except Exception as e:
            error(f"{name} failed: {e}")
            return (name, [], [])

    with phase_progress(f"Running all search providers in parallel (last {search_days} days)"):
        results = await asyncio.gather(
            *[run_provider(name, func) for name, func in providers]
        )

    all_official = []
    all_trusted = []
    for _, official, trusted in results:
        all_official.extend(official)
        all_trusted.extend(trusted)

    raw_total = len(all_official) + len(all_trusted)

    all_official = _dedup_by_url(all_official)
    all_trusted = _dedup_by_url(all_trusted)
    after_url = len(all_official) + len(all_trusted)

    if len(all_official) > 1:
        all_official = _dedup_by_similarity(all_official)
    if len(all_trusted) > 1:
        all_trusted = _dedup_by_similarity(all_trusted)

    entity = state.get("primary_entity", "")
    if entity and all_trusted:
        before_filter = len(all_trusted)
        all_trusted = [
            a for a in all_trusted
            if is_entity_relevant(entity, a.title or "", a.snippet or "")
        ]
        removed = before_filter - len(all_trusted)
        if removed:
            info(f"Entity filter '{entity}': removed {removed} irrelevant trusted article(s) ({before_filter} → {len(all_trusted)})")
        else:
            info(f"Entity filter '{entity}': all {before_filter} trusted article(s) are relevant ✓")

    final_total = len(all_official) + len(all_trusted)
    merge_summary(raw_total, after_url, final_total)

    all_combined = all_official + all_trusted
    if all_combined:
        article_table(all_combined, f"📰 Combined Article Pool (last {search_days} days)")

    next_state = dict(state)
    next_state["search_days_used"] = search_days
    next_state["official_sources"] = all_official
    next_state["trusted_sources"] = all_trusted
    return next_state


async def multi_search_agent(state: ResearchState) -> ResearchState:
    """
    Multi-Source Search Aggregator.
    Runs ALL available search providers in parallel, merges, and deduplicates.
    """
    providers = _get_available_providers()

    if not providers:
        error("No search API keys configured! Cannot search.")
        state["error"] = "No search API keys available."
        return state

    for search_days in _candidate_search_windows(state.get("search_days_used") or 15):
        next_state = await _run_provider_pass(state, search_days)
        if next_state.get("official_sources") or next_state.get("trusted_sources"):
            return next_state

    state["search_days_used"] = _candidate_search_windows(state.get("search_days_used") or 15)[-1]
    state["official_sources"] = []
    state["trusted_sources"] = []
    return state
