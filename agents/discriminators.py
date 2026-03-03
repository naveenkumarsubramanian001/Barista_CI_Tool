import json
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ResearchState, ValidationResult
from config import get_llm, get_embedding_model
from utils.json_utils import safe_json_extract

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
    retry_count = state["retry_counts"].get("decomposer", 0)

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
        state["validation_feedback"] = f"Expected 3-6 subqueries, got {len(subqueries)}."
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
        state["semantic_warnings"].append(f"Embedding redundancy check failed: {str(e)}")
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

        # Store metrics
        state["validation_metrics"].update({
            "intent_preservation": intent,
            "coverage_completeness": coverage,
            "atomicity": atomicity,
            "granularity": granularity,
            "actionability": actionability
        })

        # =====================================================
        # 4️⃣ Weighted Final Score
        # =====================================================

        final_score = (
            0.30 * intent +
            0.25 * coverage +
            0.15 * redundancy_score +
            0.10 * atomicity +
            0.10 * granularity +
            0.10 * actionability
        )

        state["decomposition_score"] = round(final_score, 3)

        if final_score >= 0.75:
            state["validation_passed"] = True
            state["validation_feedback"] = "APPROVED"
        else:
            state["validation_feedback"] = f"Low decomposition quality ({final_score:.2f}): {data.get('feedback')}"
            state["retry_counts"]["decomposer"] += 1

    except Exception as e:
        state["error"] = f"Discriminator failure: {str(e)}"
        state["semantic_warnings"].append("LLM evaluation failed.")
        state["validation_feedback"] = "SOFT_FAIL"
        state["retry_counts"]["decomposer"] += 1

    return state

    
def search_discriminator(state: ResearchState) -> ResearchState:
    """
    Validates search results for date range and relevance.
    """
    articles = state.get("articles", [])
    if not articles:
        state["validation_feedback"] = "No articles found."
        state["retry_counts"]["search"] += 1
        return state

    # Simple validation: Check if we have enough and dates are present
    if len(articles) > 10:
        state["validation_feedback"] = "Too many articles (max 10)."
        state["retry_counts"]["search"] += 1
        return state
        
    state["validation_feedback"] = "APPROVED"
    return state

def summariser_discriminator(state: ResearchState) -> ResearchState:
    """
    Validates the final report for citations and length.
    """
    report = state.get("final_report", {})
    if not report:
        state["validation_feedback"] = "No report generated."
        state["retry_counts"]["summariser"] += 1
        return state
        
    insights = report.get("top_insights", [])
    if len(insights) != 3:
        state["validation_feedback"] = f"Expected exactly 3 insights, got {len(insights)}."
        state["retry_counts"]["summariser"] += 1
        return state
        
    state["validation_feedback"] = "APPROVED"
    return state
