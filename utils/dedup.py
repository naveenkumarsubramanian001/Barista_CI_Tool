from typing import List
from models.schemas import VerifiedArticle


def deduplicate_articles(articles: List[VerifiedArticle]) -> List[VerifiedArticle]:
    seen_urls = set()
    unique_articles = []
    for article in articles:
        if article.url not in seen_urls:
            unique_articles.append(article)
            seen_urls.add(article.url)
    return unique_articles
