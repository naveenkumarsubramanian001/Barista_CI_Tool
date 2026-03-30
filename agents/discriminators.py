"""
Research-Grade Discriminators (Hybrid Fuzzy + Weighted Average) with Rich logging.
"""

import json
import asyncio
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ResearchState, ValidationResult
from config import get_llm, get_embedding_model
from utils.json_utils import safe_json_extract
from agents.fuzzy_discriminator import (
    compute_hybrid_score,
    compute_recency_score,
    ensure_minimum_sources,
)
from utils.logger import (
    console, section, info, success, warning, error,
    detail, step, score_table, phase_progress
)


def decomposer_discriminator(state: ResearchState) -> ResearchState:
    """Research-Grade Decomposition Validator."""
    section("Decomposer Validation", "🧪")
    
    original_query = state.get("original_query", "")
    subqueries = state.get("subqueries", [])
    retry_count = state["retry_counts"].get("decomposer", 0)

    state["validation_metrics"] = {}
    state["redundancy_pairs"] = []
    state["coverage_gaps"] = []
    state["semantic_warnings"] = []
    state["validation_passed"] = False
    state["decomposition_score"] = 0.0

    # Structural Checks
    if not isinstance(subqueries, list) or not (3 <= len(subqueries) <= 6):
        warning(f"Expected 3-6 subqueries, got {len(subqueries)}")
        state["validation_feedback"] = f"Expected 3-6 subqueries, got {len(subqueries)}."
        state["retry_counts"]["decomposer"] += 1
        return state

    if not all(isinstance(q, str) and q.strip() for q in subqueries):
        warning("All subqueries must be non-empty strings")
        state["validation_feedback"] = "All subqueries must be non-empty strings."
        state["retry_counts"]["decomposer"] += 1
        return state

    if len(set(subqueries)) != len(subqueries):
        warning("Exact duplicate subqueries detected")
        state["validation_feedback"] = "Exact duplicate subqueries detected."
        state["retry_counts"]["decomposer"] += 1
        return state

    # Embedding Redundancy Detection
    step(1, 3, "Embedding-based redundancy detection")
    try:
        embed_model = get_embedding_model()
        embeddings = np.array(embed_model.embed_documents(subqueries))
        similarity_matrix = cosine_similarity(embeddings)
        redundancy_penalty = 0

        for i, j in itertools.combinations(range(len(subqueries)), 2):
            if similarity_matrix[i][j] > 0.85:
                state["redundancy_pairs"].append([subqueries[i], subqueries[j]])
                redundancy_penalty += 1
                detail(f"Redundant pair (sim={similarity_matrix[i][j]:.3f}): '{subqueries[i][:40]}' ↔ '{subqueries[j][:40]}'")

        redundancy_score = 1.0 - min(redundancy_penalty * 0.2, 1.0)
        state["validation_metrics"]["redundancy_score"] = redundancy_score
    except Exception as e:
        warning(f"Embedding redundancy check failed: {e}")
        redundancy_score = 0.7

    # LLM Semantic Evaluation
    step(2, 3, "LLM semantic evaluation")
    prompt = ChatPromptTemplate.from_template("""
    Return ONLY valid JSON.
    You are an expert research evaluator. Evaluate the decomposition quality.
    Original Query: {original_query}
    Subqueries: {subqueries}

    Score each from 0.0 to 1.0:
    - intent_preservation, coverage_completeness, atomicity, granularity, actionability

    Also list missing aspects and feedback.

    Return JSON:
    {{"intent_preservation": float, "coverage_completeness": float, "atomicity": float,
      "granularity": float, "actionability": float, "missing_aspects": [], "feedback": "short explanation"}}
    """)

    chain = prompt | get_llm()

    try:
        response = chain.invoke({
            "original_query": original_query,
            "subqueries": json.dumps(subqueries)
        })
        data = safe_json_extract(response.content)

        intent = data["intent_preservation"]
        coverage = data["coverage_completeness"]
        atomicity = data["atomicity"]
        granularity = data["granularity"]
        actionability = data["actionability"]
        state["coverage_gaps"] = data.get("missing_aspects", [])

        state["validation_metrics"].update({
            "intent_preservation": intent, "coverage_completeness": coverage,
            "atomicity": atomicity, "granularity": granularity, "actionability": actionability
        })

        final_score = (
            0.30 * intent + 0.25 * coverage + 0.15 * redundancy_score +
            0.10 * atomicity + 0.10 * granularity + 0.10 * actionability
        )
        state["decomposition_score"] = round(final_score, 3)

        step(3, 3, f"Final score: {final_score:.3f}")

        if final_score >= 0.75:
            state["validation_passed"] = True
            state["validation_feedback"] = "APPROVED"
            success(f"Decomposition APPROVED (score={final_score:.3f})")
        else:
            state["validation_feedback"] = f"Low decomposition quality ({final_score:.2f}): {data.get('feedback')}"
            warning(f"Decomposition REJECTED (score={final_score:.3f})")
            state["retry_counts"]["decomposer"] += 1

    except Exception as e:
        state["error"] = f"Discriminator failure: {str(e)}"
        state["semantic_warnings"].append("LLM evaluation failed.")
        state["validation_feedback"] = "SOFT_FAIL"
        error(f"LLM evaluation failed: {e}")
        state["retry_counts"]["decomposer"] += 1

    return state


def search_discriminator(state: ResearchState) -> ResearchState:
    """Hybrid Fuzzy + Weighted Average Search Result Discriminator."""
    section("Search Discriminator (Hybrid Fuzzy)", "🧠")

    official_sources = state.get("official_sources", [])
    trusted_sources = state.get("trusted_sources", [])
    articles = official_sources + trusted_sources
    original_query = state.get("original_query", "")
    subqueries = state.get("subqueries", [])

    if not articles:
        warning("No articles found")
        state["validation_feedback"] = "No articles found."
        state["retry_counts"]["search"] += 1
        return state

    if len(articles) < 2:
        warning("Too few articles to discriminate")
        state["validation_feedback"] = "Too few articles to discriminate. Need at least 2."
        state["retry_counts"]["search"] += 1
        return state

    info(f"Evaluating {len(articles)} articles ({len(official_sources)} official + {len(trusted_sources)} trusted)")

    # Embedding Relevance
    step(1, 4, "Computing embedding relevance scores")
    relevance_scores = {}
    try:
        embed_model = get_embedding_model()
        query_text = original_query + " " + " ".join(subqueries)
        query_embedding = np.array(embed_model.embed_documents([query_text]))
        article_texts = [f"{a.title} {a.snippet}" for a in articles]
        article_embeddings = np.array(embed_model.embed_documents(article_texts))
        similarities = cosine_similarity(query_embedding, article_embeddings)[0]
        for i, article in enumerate(articles):
            relevance_scores[article.url] = float(similarities[i])
    except Exception as e:
        warning(f"Embedding relevance check failed: {e}")
        for article in articles:
            relevance_scores[article.url] = 0.5

    # LLM Evaluation
    step(2, 4, "Running LLM quality evaluation")
    articles_for_eval = []
    for i, article in enumerate(articles):
        articles_for_eval.append({
            "index": i, "title": article.title, "url": article.url,
            "snippet": article.snippet[:500], "published_date": article.published_date
        })

    prompt = ChatPromptTemplate.from_template("""
    Return ONLY valid JSON.
    You are an expert research quality evaluator.
    Original Query: {original_query}

    Articles to evaluate:
    {articles_json}

    For EACH article, score (0.0 to 1.0):
    - source_credibility, content_relevance, information_quality, recency_value
    Also flag is_duplicate (bool) and is_low_quality (bool).

    Return JSON:
    {{"evaluations": [{{"index": int, "source_credibility": float, "content_relevance": float,
      "information_quality": float, "recency_value": float, "is_duplicate": bool,
      "is_low_quality": bool, "reason": "one-line"}}],
      "overall_feedback": "brief assessment"}}
    """)

    chain = prompt | get_llm()
    llm_scores = {}

    try:
        response = chain.invoke({
            "original_query": original_query,
            "articles_json": json.dumps(articles_for_eval, indent=2)
        })
        data = safe_json_extract(response.content)
        for ev in data.get("evaluations", []):
            idx = ev.get("index")
            if idx is not None and 0 <= idx < len(articles):
                url = articles[idx].url
                llm_scores[url] = {
                    "credibility": float(ev.get("source_credibility", 0.5)),
                    "relevance": float(ev.get("content_relevance", 0.5)),
                    "quality": float(ev.get("information_quality", 0.5)),
                    "recency": float(ev.get("recency_value", 0.5)),
                    "is_duplicate": bool(ev.get("is_duplicate", False)),
                    "is_low_quality": bool(ev.get("is_low_quality", False)),
                    "reason": ev.get("reason", "")
                }
    except Exception as e:
        warning(f"LLM evaluation failed: {e}")
        for article in articles:
            llm_scores[article.url] = {
                "credibility": 0.5, "relevance": 0.5, "quality": 0.5, "recency": 0.5,
                "is_duplicate": False, "is_low_quality": False, "reason": "LLM unavailable"
            }

    # Hybrid Scoring
    step(3, 4, "Computing hybrid fuzzy + weighted scores")
    scored_articles = []
    score_display = []

    for article in articles:
        url = article.url
        embedding_relevance = relevance_scores.get(url, 0.5)
        llm_eval = llm_scores.get(url, {})

        if llm_eval.get("is_duplicate") or llm_eval.get("is_low_quality"):
            detail(f"Hard reject: {article.title[:50]} — {llm_eval.get('reason', '')}")
            score_display.append((0, 0, 0, article.title, False))
            continue

        recency_val = compute_recency_score(article.published_date)
        combined_relevance = 0.5 * embedding_relevance + 0.5 * llm_eval.get("relevance", 0.5)

        hybrid_score, fuzzy_score, weighted_score = compute_hybrid_score(
            relevance=combined_relevance,
            credibility=llm_eval.get("credibility", 0.5),
            quality=llm_eval.get("quality", 0.5),
            recency=recency_val,
        )

        article.score = hybrid_score
        scored_articles.append((hybrid_score, article))
        score_display.append((hybrid_score, fuzzy_score, weighted_score, article.title, hybrid_score >= 0.35))

    # Display scoring table
    if score_display:
        score_table(score_display)

    scored_articles.sort(key=lambda x: x[0], reverse=True)

    # Min 3 Sources
    step(4, 4, "Applying minimum 3 source guarantee")
    filtered = ensure_minimum_sources(scored_articles, min_count=3, initial_threshold=0.35, floor_threshold=0.20)

    official_filtered = [a for a in filtered if a.source_type == "official"]
    trusted_filtered = [a for a in filtered if a.source_type == "trusted"]

    success(f"Passed: {len(official_filtered)} official + {len(trusted_filtered)} trusted = {len(filtered)} total")

    if not official_filtered and not trusted_filtered:
        error("All articles scored below quality threshold")
        state["validation_feedback"] = "All articles scored below quality threshold. Retry search."
        state["retry_counts"]["search"] += 1
        return state

    state["official_sources"] = official_filtered
    state["trusted_sources"] = trusted_filtered
    state["validation_feedback"] = "APPROVED"
    return state


def summariser_discriminator(state: ResearchState) -> ResearchState:
    """Validates the final report."""
    section("Report Validation", "📝")

    report = state.get("final_report", {})
    if not report:
        error("No report generated")
        state["validation_feedback"] = "No report generated."
        state["retry_counts"]["summariser"] += 1
        return state

    official_insights = report.get("official_insights", [])
    trusted_insights = report.get("trusted_insights", [])
    executive_summary = report.get("executive_summary", "")
    key_findings = report.get("key_findings", [])

    if not official_insights and not trusted_insights:
        warning("Expected at least one insight, got 0")
        state["validation_feedback"] = "Expected at least one insight, got 0."
        state["retry_counts"]["summariser"] += 1
        return state

    if not executive_summary or len(executive_summary) < 50:
        warning("Executive summary is too short or missing")
        state["validation_feedback"] = "Executive summary is too short or missing."
        state["retry_counts"]["summariser"] += 1
        return state

    success(f"Report validated: {len(key_findings)} findings, {len(official_insights)} official + {len(trusted_insights)} trusted insights")
    state["validation_feedback"] = "APPROVED"
    return state
