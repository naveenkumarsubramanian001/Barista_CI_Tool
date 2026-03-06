def build_site_query(subquery: str, domains: list[str]) -> str:
    """
    Builds a site-restricted search query.
    Example: site:apple.com OR site:samsung.com subquery
    """
    if not domains:
        return subquery
    
    # Limit to 10 domains to avoid too long queries
    limited_domains = domains[:10]
    site_prefix = " OR ".join([f"site:{domain}" for domain in limited_domains])
    return f"{site_prefix} {subquery}"
