"""
Rank and filter node with Rich logging.
"""

from models.schemas import ResearchState
from utils.logger import section, info, success, article_table


def rank_filter_node(state: ResearchState) -> ResearchState:
    """Ranks articles using fuzzy scores + keyword overlap tiebreaker."""
    section("Rank & Filter", "📊")

    official_sources = state.get("official_sources", [])
    trusted_sources = state.get("trusted_sources", [])
    query = state.get("original_query", "").lower()

    if not official_sources and not trusted_sources:
        info("No articles to rank")
        return state

    query_terms = set(query.split())

    def rank_articles(articles_list):
        scored = []
        for article in articles_list:
            fuzzy_score = getattr(article, "score", 0.0) or 0.0
            text = (article.title + " " + article.snippet).lower()
            keyword_score = sum(1 for term in query_terms if term in text) / max(len(query_terms), 1)
            combined = fuzzy_score * 100 + keyword_score
            scored.append((combined, article))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = []
        for i, (_, article) in enumerate(scored):
            # Top 3 are auto-selected defaults in the review UI.
            article.priority = i < 3
            ranked.append(article)
        return ranked

    final_output = {
        "official_sources": rank_articles(official_sources),
        "trusted_sources": rank_articles(trusted_sources)
    }

    state["final_ranked_output"] = final_output

    all_ranked = final_output["official_sources"] + final_output["trusted_sources"]
    if all_ranked:
        article_table(all_ranked, "🏆 Top Ranked Articles")

    success(f"Ranked: {len(final_output['official_sources'])} official + {len(final_output['trusted_sources'])} trusted")
    return state
