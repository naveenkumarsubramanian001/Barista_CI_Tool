from models.schemas import ResearchState


def rank_filter_node(state: ResearchState) -> ResearchState:
    """
    Pure Python node to rank and select top 3 articles.
    """
    from models.schemas import Article
    official_sources = [Article(**a) if isinstance(a, dict) else a for a in state.get("official_sources", [])]
    trusted_sources = [Article(**a) if isinstance(a, dict) else a for a in state.get("trusted_sources", [])]
    query = state.get("original_query", "").lower()

    if not official_sources and not trusted_sources:
        return state

    state.setdefault("logs", [])
    state["logs"].append("⚖️ Scoring and filtering articles for relevance...")
    query_terms = set(query.split())

    def rank_articles(articles_list):
        # Scoring: Keyword overlap in title/snippet + recency (if possible)
        scored = []
        for article in articles_list:
            text = (article.title + " " + article.snippet).lower()
            score = sum(1 for term in query_terms if term in text)
            scored.append((score, article))

        # Sort by score desc
        scored.sort(key=lambda x: x[0], reverse=True)

        ranked_articles = []
        for i, (_, article) in enumerate(scored):
            if i < 3:
                article.priority = True
            ranked_articles.append(article)
        return ranked_articles

    final_output = {
        "official_sources": [a.model_dump() for a in rank_articles(official_sources)],
        "trusted_sources": [a.model_dump() for a in rank_articles(trusted_sources)],
    }

    state["final_ranked_output"] = final_output
    return state
