"""Quick smoke test for the search_discriminator in isolation."""

from models.schemas import Article
from agents.discriminators import search_discriminator

# Simulated state with fake articles
test_state = {
    "original_query": "smartphone competitors under 10000 INR latest advancements",
    "subqueries": [
        "budget smartphones under 10000 INR 2025",
        "Realme vs Redmi budget segment news",
    ],
    "articles": [
        Article(
            title="Redmi Note 14 launched at Rs 9,999",
            url="https://gsmarena.com/redmi-note-14",
            snippet="Xiaomi launched the Redmi Note 14 in India at a starting price of Rs 9,999, featuring a Snapdragon 4 Gen 2 processor and 5000mAh battery.",
            published_date="2025-12-01",
        ),
        Article(
            title="Realme Narzo 70 Pro review",
            url="https://androidauthority.com/realme-narzo-70",
            snippet="The Realme Narzo 70 Pro offers excellent value with a MediaTek Dimensity 7050 and 120Hz AMOLED display under Rs 10,000.",
            published_date="2025-11-15",
        ),
        Article(
            title="Top 10 clickbait phones you WONT believe",
            url="https://randomspamblog.xyz/top10",
            snippet="You won't believe these phones exist! Click here to see the most amazing deals ever!!!",
            published_date="2025-10-01",
        ),
        Article(
            title="Samsung Galaxy M15 5G specs and price",
            url="https://theverge.com/samsung-m15",
            snippet="Samsung's Galaxy M15 5G enters the budget segment in India with a 6000mAh battery and Exynos 1330 chipset at Rs 9,499.",
            published_date="2025-11-20",
        ),
        Article(
            title="Best budget phones December 2025",
            url="https://techcrunch.com/budget-phones-dec",
            snippet="Our curated list of the best budget smartphones under Rs 10,000 for December 2025 including Redmi, Realme, and Samsung models.",
            published_date="2025-12-05",
        ),
    ],
    "retry_counts": {"decomposer": 0, "search": 0, "summariser": 0},
    "validation_feedback": "",
    "validation_passed": False,
    "validation_metrics": {},
    "decomposition_score": 0.0,
    "redundancy_pairs": [],
    "coverage_gaps": [],
    "semantic_warnings": [],
    "error": None,
}

print("=" * 60)
print("TESTING search_discriminator")
print("=" * 60)
print(f"Input: {len(test_state['articles'])} articles\n")

result = search_discriminator(test_state)

print(f"\nValidation feedback: {result['validation_feedback']}")
print(f"Articles after filtering: {len(result['articles'])}")
print("\nSurviving articles:")
for i, a in enumerate(result["articles"], 1):
    print(f"  [{i}] {a.title}  —  {a.url}")
