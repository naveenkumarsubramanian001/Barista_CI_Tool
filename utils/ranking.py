from typing import List
from models.schemas import VerifiedArticle


def rank_articles(articles: List[VerifiedArticle], query: str) -> List[VerifiedArticle]:
    query_terms = query.lower().split()
    for article in articles:
        score = 0
        text = (article.title + " " + article.snippet).lower()
        for term in query_terms:
            if term in text:
                score += 1
        article.relevance_score = score
    sorted_articles = sorted(articles, key=lambda x: x.relevance_score, reverse=True)
    for i in range(min(3, len(sorted_articles))):
        sorted_articles[i].priority = True
    return sorted_articles
