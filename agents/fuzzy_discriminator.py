"""
Hybrid Fuzzy + Weighted Average Discriminator Engine.

Uses a Mamdani fuzzy inference system (scikit-fuzzy) combined with
traditional weighted average for robust, unbiased source quality scoring.

The hybrid approach:
  - Fuzzy score (60% weight): handles uncertainty, partial truth, gradual transitions
  - Weighted average (40% weight): provides deterministic baseline
  - Both must agree to reject an article (consensus rejection)
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Dict, Tuple


def _build_fuzzy_system() -> ctrl.ControlSystemSimulation:
    """
    Build and return a Mamdani fuzzy inference system for source quality.
    
    Inputs: relevance, credibility, quality, recency (all 0.0–1.0)
    Output: source_quality (0.0–1.0)
    
    The knowledge base is designed to be UNBIASED:
    - No single dimension can auto-pass an article
    - High relevance alone isn't enough (need quality too)
    - High credibility alone isn't enough (need relevance too)
    - Fresh content doesn't auto-pass (still needs quality)
    """
    universe = np.arange(0, 1.01, 0.01)

    # --- Antecedents (Inputs) ---
    relevance = ctrl.Antecedent(universe, "relevance")
    credibility = ctrl.Antecedent(universe, "credibility")
    quality = ctrl.Antecedent(universe, "quality")
    recency = ctrl.Antecedent(universe, "recency")

    # --- Consequent (Output) ---
    source_quality = ctrl.Consequent(universe, "source_quality")

    # --- Membership Functions (Inputs) ---
    # Relevance: how well article matches the query
    relevance["poor"] = fuzz.trapmf(universe, [0, 0, 0.2, 0.35])
    relevance["fair"] = fuzz.trimf(universe, [0.2, 0.4, 0.6])
    relevance["good"] = fuzz.trimf(universe, [0.5, 0.7, 0.85])
    relevance["excellent"] = fuzz.trapmf(universe, [0.75, 0.88, 1.0, 1.0])

    # Credibility: source authority / reputation
    credibility["low"] = fuzz.trapmf(universe, [0, 0, 0.25, 0.4])
    credibility["medium"] = fuzz.trimf(universe, [0.3, 0.5, 0.7])
    credibility["high"] = fuzz.trapmf(universe, [0.6, 0.8, 1.0, 1.0])

    # Quality: content substance, factuality
    quality["low"] = fuzz.trapmf(universe, [0, 0, 0.25, 0.4])
    quality["medium"] = fuzz.trimf(universe, [0.3, 0.5, 0.7])
    quality["high"] = fuzz.trapmf(universe, [0.6, 0.8, 1.0, 1.0])

    # Recency: how fresh the content is
    recency["stale"] = fuzz.trapmf(universe, [0, 0, 0.2, 0.4])
    recency["recent"] = fuzz.trimf(universe, [0.3, 0.55, 0.8])
    recency["fresh"] = fuzz.trapmf(universe, [0.7, 0.85, 1.0, 1.0])

    # --- Output Membership Functions ---
    source_quality["reject"] = fuzz.trapmf(universe, [0, 0, 0.15, 0.3])
    source_quality["borderline"] = fuzz.trimf(universe, [0.2, 0.38, 0.55])
    source_quality["acceptable"] = fuzz.trimf(universe, [0.45, 0.65, 0.82])
    source_quality["excellent"] = fuzz.trapmf(universe, [0.72, 0.85, 1.0, 1.0])

    # --- Fuzzy Rules (Unbiased Knowledge Base) ---
    # Strong positive signals — need MULTIPLE high dimensions
    rule1 = ctrl.Rule(relevance["excellent"] & credibility["high"] & quality["high"], source_quality["excellent"])
    rule2 = ctrl.Rule(relevance["excellent"] & credibility["high"] & quality["medium"], source_quality["excellent"])
    rule3 = ctrl.Rule(relevance["good"] & credibility["high"] & quality["high"], source_quality["excellent"])
    
    # Good combinations — two strong + one moderate
    rule4 = ctrl.Rule(relevance["excellent"] & credibility["medium"] & quality["high"], source_quality["acceptable"])
    rule5 = ctrl.Rule(relevance["good"] & credibility["high"] & quality["medium"], source_quality["acceptable"])
    rule6 = ctrl.Rule(relevance["good"] & credibility["medium"] & quality["high"], source_quality["acceptable"])
    rule7 = ctrl.Rule(relevance["excellent"] & credibility["medium"] & quality["medium"], source_quality["acceptable"])
    
    # Recency boosts
    rule8 = ctrl.Rule(relevance["good"] & quality["medium"] & recency["fresh"], source_quality["acceptable"])
    rule9 = ctrl.Rule(relevance["fair"] & credibility["high"] & recency["fresh"], source_quality["acceptable"])
    
    # Borderline cases — mixed signals
    rule10 = ctrl.Rule(relevance["fair"] & credibility["medium"] & quality["medium"], source_quality["borderline"])
    rule11 = ctrl.Rule(relevance["good"] & credibility["low"] & quality["medium"], source_quality["borderline"])
    rule12 = ctrl.Rule(relevance["fair"] & credibility["high"] & quality["low"], source_quality["borderline"])
    rule13 = ctrl.Rule(relevance["good"] & credibility["medium"] & quality["low"], source_quality["borderline"])
    
    # Rejection rules — need consensus (multiple bad signals)
    rule14 = ctrl.Rule(relevance["poor"] & credibility["low"], source_quality["reject"])
    rule15 = ctrl.Rule(relevance["poor"] & quality["low"], source_quality["reject"])
    rule16 = ctrl.Rule(credibility["low"] & quality["low"], source_quality["reject"])
    rule17 = ctrl.Rule(relevance["poor"] & credibility["medium"] & quality["low"], source_quality["reject"])
    
    # Stale content penalty 
    rule18 = ctrl.Rule(recency["stale"] & relevance["fair"] & quality["low"], source_quality["reject"])
    rule19 = ctrl.Rule(recency["stale"] & credibility["low"], source_quality["reject"])
    
    # Fair relevance with decent quality — give a chance
    rule20 = ctrl.Rule(relevance["fair"] & credibility["medium"] & quality["high"], source_quality["acceptable"])

    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
             rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


# Module-level singleton (built once)
_fuzzy_sim = None


def _get_fuzzy_sim() -> ctrl.ControlSystemSimulation:
    global _fuzzy_sim
    if _fuzzy_sim is None:
        _fuzzy_sim = _build_fuzzy_system()
    return _fuzzy_sim


def compute_fuzzy_score(
    relevance: float,
    credibility: float, 
    quality: float,
    recency: float
) -> float:
    """
    Compute the fuzzy source quality score.
    
    Args:
        relevance: 0.0-1.0 (embedding similarity)
        credibility: 0.0-1.0 (LLM source assessment)
        quality: 0.0-1.0 (LLM content quality)
        recency: 0.0-1.0 (date-based freshness)
    
    Returns:
        source_quality: 0.0-1.0 (defuzzified via centroid)
    """
    try:
        sim = _get_fuzzy_sim()
        # Clamp inputs to valid range
        sim.input["relevance"] = max(0.01, min(0.99, relevance))
        sim.input["credibility"] = max(0.01, min(0.99, credibility))
        sim.input["quality"] = max(0.01, min(0.99, quality))
        sim.input["recency"] = max(0.01, min(0.99, recency))
        sim.compute()
        return float(sim.output["source_quality"])
    except Exception as e:
        print(f"   ⚠️ Fuzzy computation failed: {e} — using weighted fallback")
        # Fallback to weighted average if fuzzy fails
        return compute_weighted_score(relevance, credibility, quality, recency)


def compute_weighted_score(
    relevance: float,
    credibility: float,
    quality: float,
    recency: float
) -> float:
    """
    Traditional weighted average score (deterministic baseline).
    """
    return (
        0.30 * relevance +
        0.25 * credibility +
        0.25 * quality +
        0.20 * recency
    )


def compute_hybrid_score(
    relevance: float,
    credibility: float,
    quality: float,
    recency: float
) -> Tuple[float, float, float]:
    """
    Hybrid scoring: 60% fuzzy + 40% weighted average.
    
    Returns:
        (hybrid_score, fuzzy_score, weighted_score)
    """
    fuzzy_score = compute_fuzzy_score(relevance, credibility, quality, recency)
    weighted_score = compute_weighted_score(relevance, credibility, quality, recency)
    
    hybrid = 0.60 * fuzzy_score + 0.40 * weighted_score
    return (round(hybrid, 4), round(fuzzy_score, 4), round(weighted_score, 4))


def compute_recency_score(published_date: str, max_days: int = 180) -> float:
    """
    Convert a date string to a 0.0-1.0 recency score.
    1.0 = today, 0.0 = older than max_days.
    """
    from datetime import datetime
    if not published_date:
        return 0.5  # Unknown date gets neutral score

    try:
        pub = datetime.strptime(published_date[:10], "%Y-%m-%d")
        delta = (datetime.now() - pub).days
        if delta < 0:
            return 1.0
        if delta > max_days:
            return 0.0
        return round(1.0 - (delta / max_days), 3)
    except (ValueError, TypeError):
        return 0.5


def ensure_minimum_sources(
    scored_articles: list,
    min_count: int = 3,
    initial_threshold: float = 0.35,
    floor_threshold: float = 0.20,
    step: float = 0.05
) -> list:
    """
    Guarantee at least min_count sources pass.
    Progressively lowers threshold until enough articles pass.
    
    Args:
        scored_articles: list of (score, article) tuples, sorted desc
        min_count: minimum number of articles to keep
        initial_threshold: starting quality threshold
        floor_threshold: absolute minimum threshold  
        step: threshold reduction step
        
    Returns:
        List of articles that pass the threshold, guaranteed >= min_count if available
    """
    threshold = initial_threshold
    
    while threshold >= floor_threshold:
        passing = [art for score, art in scored_articles if score >= threshold]
        if len(passing) >= min_count:
            return passing
        threshold -= step
    
    # If even floor threshold doesn't give enough, return whatever we have
    # sorted by score (best first), up to available count
    return [art for _, art in scored_articles[:max(min_count, len(scored_articles))]]
