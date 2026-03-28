import json
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ResearchState
from config import get_llm, get_embedding_model, ENABLE_FUZZY_SCORING
from utils.json_utils import safe_json_extract

try:
    from agents.fuzzy_discriminator import (
        compute_hybrid_score,
        compute_recency_score,
        ensure_minimum_sources,
    )
    FUZZY_AVAILABLE = True
except Exception:
    FUZZY_AVAILABLE = False


def decomposer_discriminator(state: ResearchState) -> ResearchState:
    """
    Research-Grade Decomposition Validator
    - Structural validation
    - Embedding-based redundancy detection
    - LLM-based coverage + atomicity scoring
    - Weighted scoring system
    """

    original_query = state.get("original_query", "")
    subqueries = state.get("subqueries", [])
    state["retry_counts"].get("decomposer", 0)

    state["validation_metrics"] = {}
    state["redundancy_pairs"] = []
    state["coverage_gaps"] = []
    state["semantic_warnings"] = []
    state["validation_passed"] = False
    state["decomposition_score"] = 0.0

    # =====================================================
    # 1️⃣ Deterministic Structural Checks
    # =====================================================

    if not isinstance(subqueries, list) or not (3 <= len(subqueries) <= 6):
        state["validation_feedback"] = (
            f"Expected 3-6 subqueries, got {len(subqueries)}."
        )
        state["retry_counts"]["decomposer"] += 1
        return state

    if not all(isinstance(q, str) and q.strip() for q in subqueries):
        state["validation_feedback"] = "All subqueries must be non-empty strings."
        state["retry_counts"]["decomposer"] += 1
        return state

    if len(set(subqueries)) != len(subqueries):
        state["validation_feedback"] = "Exact duplicate subqueries detected."
        state["retry_counts"]["decomposer"] += 1
        return state

    # =====================================================
    # 2️⃣ Embedding-Based Semantic Redundancy Detection
    # =====================================================

    try:
        embed_model = get_embedding_model()
        embeddings = np.array(embed_model.embed_documents(subqueries))

        similarity_matrix = cosine_similarity(embeddings)
        redundancy_penalty = 0

        for i, j in itertools.combinations(range(len(subqueries)), 2):
            if similarity_matrix[i][j] > 0.85:
                state["redundancy_pairs"].append([subqueries[i], subqueries[j]])
                redundancy_penalty += 1

        redundancy_score = 1.0 - min(redundancy_penalty * 0.2, 1.0)
        state["validation_metrics"]["redundancy_score"] = redundancy_score

    except Exception as e:
        state["semantic_warnings"].append(
            f"Embedding redundancy check failed: {str(e)}"
        )
        redundancy_score = 0.7  # soft fallback

    # =====================================================
    # 3️⃣ LLM Semantic Evaluation
    # =====================================================

    prompt = ChatPromptTemplate.from_template("""
    Return ONLY valid JSON.

    You are an expert research evaluator.
    Evaluate the decomposition quality.

    Original Query: {original_query}
    Subqueries: {subqueries}

    Score each from 0.0 to 1.0:

    - intent_preservation
    - coverage_completeness
    - atomicity
    - granularity
    - actionability

    Also:
    - List any missing aspects in coverage.
    - Mention if any subquery bundles multiple questions.

    Return JSON:
    {{
      "intent_preservation": float,
      "coverage_completeness": float,
      "atomicity": float,
      "granularity": float,
      "actionability": float,
      "missing_aspects": [],
      "feedback": "short explanation"
    }}
    """)

    chain = prompt | get_llm()

    try:
        response = chain.invoke(
            {"original_query": original_query, "subqueries": json.dumps(subqueries)}
        )

        data = safe_json_extract(response.content)

        intent = data["intent_preservation"]
        coverage = data["coverage_completeness"]
        atomicity = data["atomicity"]
        granularity = data["granularity"]
        actionability = data["actionability"]

        state["coverage_gaps"] = data.get("missing_aspects", [])

        # Store metrics
        state["validation_metrics"].update(
            {
                "intent_preservation": intent,
                "coverage_completeness": coverage,
                "atomicity": atomicity,
                "granularity": granularity,
                "actionability": actionability,
            }
        )

        # =====================================================
        # 4️⃣ Weighted Final Score
        # =====================================================

        final_score = (
            0.30 * intent
            + 0.25 * coverage
            + 0.15 * redundancy_score
            + 0.10 * atomicity
            + 0.10 * granularity
            + 0.10 * actionability
        )

        state["decomposition_score"] = round(final_score, 3)

        if final_score >= 0.75:
            state["validation_passed"] = True
            state["validation_feedback"] = "APPROVED"
        else:
            state["validation_feedback"] = (
                f"Low decomposition quality ({final_score:.2f}): {data.get('feedback')}"
            )
            state["retry_counts"]["decomposer"] += 1

    except Exception as e:
        state["error"] = f"Discriminator failure: {str(e)}"
        state["semantic_warnings"].append("LLM evaluation failed.")
        state["validation_feedback"] = "SOFT_FAIL"
        state["retry_counts"]["decomposer"] += 1

    return state


def search_discriminator(state: ResearchState) -> ResearchState:
    """
    Research-Grade Search Result Discriminator.
    Evaluates each article for relevance, source credibility, content quality,
    and recency — then filters to keep only the most accurate and reliable results.

    Pipeline:
      1. Structural checks (empty results, min threshold)
      2. Embedding-based relevance scoring (query ↔ article similarity)
      3. LLM-based accuracy & reliability evaluation per article
      4. Weighted composite score → filter top results
    """
    from models.schemas import Article
    official_sources = [Article(**a) if isinstance(a, dict) else a for a in state.get("official_sources", [])]
    trusted_sources = [Article(**a) if isinstance(a, dict) else a for a in state.get("trusted_sources", [])]
    articles = official_sources + trusted_sources
    original_query = state.get("original_query", "")
    subqueries = state.get("subqueries", [])

    # =====================================================
    # 1️⃣ Structural Checks
    # =====================================================

    if not articles:
        state["validation_feedback"] = "No articles found."
        state["retry_counts"]["search"] += 1
        return state

    if len(articles) < 2:
        state["validation_feedback"] = (
            "Too few articles to discriminate. Need at least 2."
        )
        state["retry_counts"]["search"] += 1
        return state

    # =====================================================
    # 2️⃣ Embedding-Based Relevance Scoring
    # =====================================================

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
        print(f"   ⚠️ Embedding relevance check failed: {e}")
        # Fallback: assign neutral score
        for article in articles:
            relevance_scores[article.url] = 0.5

    # =====================================================
    # 3️⃣ LLM-Based Accuracy & Reliability Evaluation
    # =====================================================

    articles_for_eval = []
    for i, article in enumerate(articles):
        articles_for_eval.append(
            {
                "index": i,
                "title": article.title,
                "url": article.url,
                "snippet": article.snippet[:500],
                "published_date": article.published_date,
            }
        )

    prompt = ChatPromptTemplate.from_template("""
    Return ONLY valid JSON.

    You are an expert research quality evaluator. Your job is to assess search results
    for ACCURACY and RELIABILITY so only the best sources are kept.

    Original Query: {original_query}

    Articles to evaluate:
    {articles_json}

    For EACH article, score these dimensions from 0.0 to 1.0:

    - source_credibility: Is the source a reputable, authoritative outlet? (e.g., major tech sites,
      official company blogs, peer-reviewed → high; unknown blogs, forums → low)
    - content_relevance: How directly does the snippet address the original query?
    - information_quality: Is the content factual, specific, and substantive (not clickbait or vague)?
    - recency_value: Is the publication date recent enough to be useful for the query?

    Also flag any article that appears to be:
    - duplicate/near-duplicate content of another article (set "is_duplicate": true)
    - clickbait or low-substance (set "is_low_quality": true)

    Return JSON:
    {{
      "evaluations": [
        {{
          "index": int,
          "source_credibility": float,
          "content_relevance": float,
          "information_quality": float,
          "recency_value": float,
          "is_duplicate": bool,
          "is_low_quality": bool,
          "reason": "one-line justification"
        }}
      ],
      "overall_feedback": "brief assessment of the result set quality"
    }}
    """)

    chain = prompt | get_llm()
    llm_scores = {}

    try:
        response = chain.invoke(
            {
                "original_query": original_query,
                "articles_json": json.dumps(articles_for_eval, indent=2),
            }
        )

        data = safe_json_extract(response.content)
        evaluations = data.get("evaluations", [])

        for ev in evaluations:
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
                    "reason": ev.get("reason", ""),
                }

    except Exception as e:
        print(f"   ⚠️ LLM evaluation failed: {e}")
        # Fallback: neutral scores for all
        for article in articles:
            llm_scores[article.url] = {
                "credibility": 0.5,
                "relevance": 0.5,
                "quality": 0.5,
                "recency": 0.5,
                "is_duplicate": False,
                "is_low_quality": False,
                "reason": "LLM evaluation unavailable",
            }

    # =====================================================
    # 4️⃣ Composite Scoring & Filtering
    # =====================================================

    scored_articles = []
    use_fuzzy = ENABLE_FUZZY_SCORING and FUZZY_AVAILABLE
    state.setdefault("logs", [])

    if ENABLE_FUZZY_SCORING and not FUZZY_AVAILABLE:
        state["logs"].append("⚠️ Fuzzy scoring requested but unavailable; using legacy scoring.")
    elif use_fuzzy:
        state["logs"].append("🧠 Fuzzy discriminator enabled for source scoring.")

    for article in articles:
        url = article.url
        embedding_relevance = relevance_scores.get(url, 0.5)
        llm_eval = llm_scores.get(url, {})

        # Skip duplicates and low-quality flagged articles
        if llm_eval.get("is_duplicate") or llm_eval.get("is_low_quality"):
            continue

        if use_fuzzy:
            recency_score = compute_recency_score(article.published_date)
            combined_relevance = 0.5 * embedding_relevance + 0.5 * llm_eval.get("relevance", 0.5)
            composite_score, fuzzy_score, weighted_score = compute_hybrid_score(
                relevance=combined_relevance,
                credibility=llm_eval.get("credibility", 0.5),
                quality=llm_eval.get("quality", 0.5),
                recency=recency_score,
            )
            state["logs"].append(
                (
                    f"🧠 Score[{article.source_type}] {article.title[:50]} | "
                    f"hybrid={composite_score:.3f}, fuzzy={fuzzy_score:.3f}, weighted={weighted_score:.3f}, "
                    f"embed={embedding_relevance:.3f}, llm_rel={llm_eval.get('relevance', 0.5):.3f}, "
                    f"cred={llm_eval.get('credibility', 0.5):.3f}, quality={llm_eval.get('quality', 0.5):.3f}, recency={recency_score:.3f}"
                )
            )
        else:
            composite_score = (
                0.25 * embedding_relevance
                + 0.25 * llm_eval.get("credibility", 0.5)
                + 0.20 * llm_eval.get("relevance", 0.5)
                + 0.20 * llm_eval.get("quality", 0.5)
                + 0.10 * llm_eval.get("recency", 0.5)
            )
            state["logs"].append(
                (
                    f"📊 Score[{article.source_type}] {article.title[:50]} | "
                    f"legacy={composite_score:.3f}, embed={embedding_relevance:.3f}, "
                    f"cred={llm_eval.get('credibility', 0.5):.3f}, rel={llm_eval.get('relevance', 0.5):.3f}, "
                    f"quality={llm_eval.get('quality', 0.5):.3f}, recency={llm_eval.get('recency', 0.5):.3f}"
                )
            )

        article.score = round(float(composite_score), 4)

        scored_articles.append((composite_score, article))

    # Sort by composite score descending
    scored_articles.sort(key=lambda x: x[0], reverse=True)

    # Keep all results above quality threshold (ranker picks top 3 for insights)
    quality_threshold = 0.35 if use_fuzzy else 0.45
    filtered = [article for score, article in scored_articles if score >= quality_threshold]

    if use_fuzzy and len(filtered) < 3:
        filtered = ensure_minimum_sources(
            scored_articles,
            min_count=3,
            initial_threshold=quality_threshold,
            floor_threshold=0.20,
        )

    # Split back into official and trusted
    official_filtered = [a for a in filtered if a.source_type == "official"]
    trusted_filtered = [a for a in filtered if a.source_type == "trusted"]

    print(
        f"   - Discriminator: {len(official_sources)} official → {len(official_filtered)} passed"
    )
    print(
        f"   - Discriminator: {len(trusted_sources)} trusted → {len(trusted_filtered)} passed"
    )

    if not official_filtered and not trusted_filtered:
        state["validation_feedback"] = (
            "All articles scored below quality threshold. Retry search."
        )
        state["retry_counts"]["search"] += 1
        return state

    state["official_sources"] = [a.model_dump() for a in official_filtered]
    state["trusted_sources"] = [a.model_dump() for a in trusted_filtered]
    state["validation_feedback"] = "APPROVED"
    return state


def summariser_discriminator(state: ResearchState) -> ResearchState:
    """
    Validates the final report for citations, length, and insight count.
    Ensures the summariser produced one insight per article.
    """
    report = state.get("final_report", {})
    if not report:
        state["validation_feedback"] = "No report generated."
        state["retry_counts"]["summariser"] += 1
        return state

    official_insights = report.get("official_insights", [])
    trusted_insights = report.get("trusted_insights", [])

    if not official_insights and not trusted_insights:
        state["validation_feedback"] = "Expected at least one insight, got 0."
        state["retry_counts"]["summariser"] += 1
        return state

    # Validate insight count matches article count that was ACTUALLY provided to summariser
    final_ranked = state.get("final_ranked_output", {})
    from models.schemas import Article
    official_sources = [Article(**a) if isinstance(a, dict) else a for a in final_ranked.get("official_sources", [])]
    trusted_sources = [Article(**a) if isinstance(a, dict) else a for a in final_ranked.get("trusted_sources", [])]
    selected_urls = state.get("selected_articles", [])

    if selected_urls:
        expected_official = len([a for a in official_sources if a.url in selected_urls])
        expected_trusted = len([a for a in trusted_sources if a.url in selected_urls])
    else:
        expected_official = len(official_sources)
        expected_trusted = len(trusted_sources)

    got_official = len(official_insights)
    got_trusted = len(trusted_insights)

    missing = []
    if expected_official > 0 and got_official < expected_official:
        missing.append(
            f"official: expected {expected_official} insights, got {got_official}"
        )
    if expected_trusted > 0 and got_trusted < expected_trusted:
        missing.append(
            f"trusted: expected {expected_trusted} insights, got {got_trusted}"
        )

    if missing:
        state["validation_feedback"] = (
            f"Insufficient insights — {'; '.join(missing)}. "
            f"Generate one insight per article."
        )
        state["retry_counts"]["summariser"] += 1
        return state

    state["validation_feedback"] = "APPROVED"
    return state
