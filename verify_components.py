"""Quick verification script for all new components."""
import sys
print("=" * 60)
print("BARISTA CI TOOL - Component Verification")
print("=" * 60)

errors = []

# 1. Test fuzzy discriminator
print("\n[1/6] Testing fuzzy_discriminator...")
try:
    from agents.fuzzy_discriminator import (
        compute_hybrid_score,
        compute_recency_score,
        ensure_minimum_sources,
    )
    h, f, w = compute_hybrid_score(0.8, 0.7, 0.6, 0.9)
    print(f"  High quality: hybrid={h:.3f} fuzzy={f:.3f} weighted={w:.3f}")
    assert h > 0.5, f"Expected high hybrid score, got {h}"
    
    h2, f2, w2 = compute_hybrid_score(0.2, 0.1, 0.15, 0.3)
    print(f"  Low quality:  hybrid={h2:.3f} fuzzy={f2:.3f} weighted={w2:.3f}")
    assert h2 < h, "Low quality should score lower than high quality"
    
    r = compute_recency_score("2026-03-23")
    print(f"  Recency (today): {r:.3f}")
    
    # Test minimum sources
    mock_articles = [(0.8, "art1"), (0.6, "art2"), (0.3, "art3"), (0.1, "art4")]
    result = ensure_minimum_sources(mock_articles, min_count=3, initial_threshold=0.5)
    assert len(result) >= 3, f"Expected >= 3 sources, got {len(result)}"
    print(f"  Min sources guarantee: {len(result)} sources (threshold lowered)")
    
    print("  OK fuzzy_discriminator")
except Exception as e:
    errors.append(f"fuzzy_discriminator: {e}")
    print(f"  FAILED: {e}")

# 2. Test schemas
print("\n[2/6] Testing models/schemas...")
try:
    from models.schemas import (
        ResearchState, Article, FinalReport, KeyFinding,
        Insight, AnalyzerState
    )
    art = Article(
        title="Test",
        url="https://example.com",
        snippet="test snippet",
        published_date="2026-03-01",
        source_type="official",
        score=0.85
    )
    assert art.score == 0.85
    
    kf = KeyFinding(
        finding_title="Test Finding",
        finding_summary="Summary text",
        source_ids=[1, 2]
    )
    assert len(kf.source_ids) == 2
    
    report = FinalReport(
        report_title="Test Report",
        executive_summary="Test summary",
        key_findings=[kf],
        official_insights=[],
        trusted_insights=[],
        cross_source_analysis="Analysis text",
        references=[]
    )
    assert report.executive_summary == "Test summary"
    
    print("  OK schemas")
except Exception as e:
    errors.append(f"schemas: {e}")
    print(f"  FAILED: {e}")

# 3. Test config
print("\n[3/6] Testing config...")
try:
    from config import (
        get_llm, TAVILY_API_KEY, BING_SEARCH_API_KEY,
        SERPER_API_KEY, SEARCH_PROVIDER
    )
    print(f"  SEARCH_PROVIDER: {SEARCH_PROVIDER}")
    print(f"  TAVILY_API_KEY: {'set' if TAVILY_API_KEY else 'empty'}")
    print(f"  BING_SEARCH_API_KEY: {'set' if BING_SEARCH_API_KEY else 'empty'}")
    print(f"  SERPER_API_KEY: {'set' if SERPER_API_KEY else 'empty'}")
    print("  OK config")
except Exception as e:
    errors.append(f"config: {e}")
    print(f"  FAILED: {e}")

# 4. Test bing_search_agent imports
print("\n[4/6] Testing agents/bing_search_agent...")
try:
    from agents.bing_search_agent import bing_search_agent, perform_single_bing_search
    print("  OK bing_search_agent imports")
except Exception as e:
    errors.append(f"bing_search_agent: {e}")
    print(f"  FAILED: {e}")

# 5. Test workflow build
print("\n[5/6] Testing graph/workflow build...")
try:
    from graph.workflow import build_graph
    graph = build_graph()
    print("  OK workflow builds")
except Exception as e:
    errors.append(f"workflow: {e}")
    print(f"  FAILED: {e}")

# 6. Test rank_filter
print("\n[6/6] Testing nodes/rank_filter...")
try:
    from nodes.rank_filter import rank_filter_node
    print("  OK rank_filter imports")
except Exception as e:
    errors.append(f"rank_filter: {e}")
    print(f"  FAILED: {e}")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"ERRORS ({len(errors)}):")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("ALL 6 COMPONENTS VERIFIED SUCCESSFULLY")
    sys.exit(0)
