from models.schemas import ResearchState

def rank_filter_node(state: ResearchState) -> ResearchState:
    """
    Pure Python node to rank and select top 3 articles.
    """
    official_sources = state.get("official_sources", [])
    trusted_sources = state.get("trusted_sources", [])
    query = state.get("original_query", "").lower()
    
    if not official_sources and not trusted_sources:
        return state
        
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
        
        top = []
        for i, (_, article) in enumerate(scored[:3]): # Top 3 per source type
            article.priority = True
            top.append(article)
        return top
        
    final_output = {
        "official_sources": rank_articles(official_sources),
        "trusted_sources": rank_articles(trusted_sources)
    }
        
    state["final_ranked_output"] = final_output
    return state
