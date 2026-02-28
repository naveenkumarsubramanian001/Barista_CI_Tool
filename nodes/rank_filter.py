from models.schemas import ResearchState

def rank_filter_node(state: ResearchState) -> ResearchState:
    """
    Pure Python node to rank and select top 3 articles.
    """
    articles = state.get("articles", [])
    query = state.get("original_query", "").lower()
    
    if not articles:
        return state
        
    # Scoring: Keyword overlap in title/snippet + recency (if possible)
    scored_articles = []
    query_terms = set(query.split())
    
    for article in articles:
        text = (article.title + " " + article.snippet).lower()
        score = sum(1 for term in query_terms if term in text)
        scored_articles.append((score, article))
        
    # Sort by score desc
    scored_articles.sort(key=lambda x: x[0], reverse=True)
    
    top_3 = []
    for i, (_, article) in enumerate(scored_articles[:3]):
        article.priority = True
        top_3.append(article)
        
    state["top_articles"] = top_3
    return state
