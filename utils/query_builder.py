def build_site_query(subquery: str, domains: list[str]) -> str:
    """
    Builds a site-restricted search query for official sources.
    Example: site:apple.com OR site:samsung.com subquery
    """
    if not domains:
        return subquery

    # Limit to 10 domains to avoid too long queries
    limited_domains = domains[:10]
    site_prefix = " OR ".join([f"site:{domain}" for domain in limited_domains])
    return f"{site_prefix} {subquery}"


def build_trusted_query(entity: str, subquery: str, domains: list[str]) -> str:
    """
    Builds an entity-anchored, site-restricted query for trusted source searches.

    Placing the quoted entity name at the front of the query forces search engines
    to prioritise articles that explicitly mention the company, eliminating broad
    topic-only results like 'AI startup funding rounds'.

    Examples:
      entity="Hugging Face", domains=["techcrunch.com", "venturebeat.com"]
        → '"Hugging Face" site:techcrunch.com OR site:venturebeat.com open-source AI'

    Falls back to build_site_query() when entity is empty (generic/multi-company
    queries where no single primary entity was detected).
    """
    if not entity:
        # No primary entity — fall back to the standard topic-only query
        return build_site_query(subquery, domains)

    if not domains:
        # No trusted domains configured — just anchor to entity + subquery
        return f'"{entity}" {subquery}'

    limited_domains = domains[:10]
    site_prefix = " OR ".join([f"site:{domain}" for domain in limited_domains])
    # Entity quoted for exact phrase match, placed before site: operators
    return f'"{entity}" {site_prefix} {subquery}'


def is_entity_relevant(entity: str, title: str, snippet: str) -> bool:
    """
    Post-retrieval relevance filter for trusted-source articles.

    Returns True if the entity name appears (case-insensitive) in either the
    article title or snippet. Returns True unconditionally when entity is empty
    so that generic queries are never silently filtered.

    This acts as a safety net after entity-anchored querying — even if a search
    engine returns a stray off-topic article, it is discarded here.

    Args:
        entity:  Primary company name, e.g. "Hugging Face". May be empty string.
        title:   Article title string.
        snippet: Article snippet / description string.

    Returns:
        bool — True means keep the article, False means discard.
    """
    if not entity:
        return True  # No entity to filter on — pass everything through

    entity_lower = entity.lower()
    combined = (title + " " + snippet).lower()
    return entity_lower in combined

