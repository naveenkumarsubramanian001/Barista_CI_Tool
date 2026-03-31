"""
Unit tests for entity-anchored trusted source search fix.

Tests cover:
  - build_trusted_query() output format and fallback
  - is_entity_relevant() filtering logic
  - primary_entity stored in state by url_discovery (monkeypatched)
  - entity filter applied in multi_search_agent
"""

import sys
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# build_trusted_query
# ---------------------------------------------------------------------------

class TestBuildTrustedQuery:

    def test_with_entity_and_domains(self):
        from utils.query_builder import build_trusted_query
        result = build_trusted_query(
            entity="Hugging Face",
            subquery="open-source AI models",
            domains=["techcrunch.com", "venturebeat.com"],
        )
        # Entity must be quoted and at the start
        assert result.startswith('"Hugging Face"')
        # Both domains embedded as site: operators
        assert "site:techcrunch.com" in result
        assert "site:venturebeat.com" in result
        # Original subquery still present
        assert "open-source AI models" in result

    def test_entity_is_exactly_quoted(self):
        """Multi-word entities must be wrapped in double quotes for exact phrase match."""
        from utils.query_builder import build_trusted_query
        result = build_trusted_query("Hugging Face", "AI", ["techcrunch.com"])
        assert '"Hugging Face"' in result, "Entity must be in double quotes"
        assert "Hugging Face" in result  # sanity check (quoted is a superset)

    def test_empty_entity_fallback_to_site_query(self):
        """When entity is empty, should produce same output as build_site_query."""
        from utils.query_builder import build_trusted_query, build_site_query
        domains = ["techcrunch.com", "venturebeat.com"]
        subquery = "AI models"
        expected = build_site_query(subquery, domains)
        result = build_trusted_query("", subquery, domains)
        assert result == expected, "Empty entity must fall back to build_site_query"

    def test_entity_with_no_domains_returns_entity_plus_subquery(self):
        from utils.query_builder import build_trusted_query
        result = build_trusted_query("OpenAI", "competitor analysis", [])
        assert '"OpenAI"' in result
        assert "competitor analysis" in result
        assert "site:" not in result, "No site: prefix when no domains provided"

    def test_domains_capped_at_10(self):
        """Should not embed more than 10 site: operators to keep query length reasonable."""
        from utils.query_builder import build_trusted_query
        domains = [f"domain{i}.com" for i in range(15)]
        result = build_trusted_query("OpenAI", "news", domains)
        site_count = result.count("site:")
        assert site_count <= 10, f"Expected ≤ 10 site: operators, got {site_count}"


# ---------------------------------------------------------------------------
# is_entity_relevant
# ---------------------------------------------------------------------------

class TestIsEntityRelevant:

    def test_entity_in_title_returns_true(self):
        from utils.query_builder import is_entity_relevant
        assert is_entity_relevant(
            entity="Hugging Face",
            title="Hugging Face raises $100M Series D",
            snippet="The AI company expanded its open-source platform.",
        ) is True

    def test_entity_in_snippet_returns_true(self):
        from utils.query_builder import is_entity_relevant
        assert is_entity_relevant(
            entity="Hugging Face",
            title="Top AI companies to watch",
            snippet="Hugging Face continues to lead in open-source transformer models.",
        ) is True

    def test_entity_absent_returns_false(self):
        from utils.query_builder import is_entity_relevant
        assert is_entity_relevant(
            entity="Hugging Face",
            title="OpenAI launches GPT-5",
            snippet="The new model sets a benchmark in reasoning tasks.",
        ) is False

    def test_case_insensitive_match(self):
        from utils.query_builder import is_entity_relevant
        assert is_entity_relevant(
            entity="Hugging Face",
            title="HUGGING FACE announces partnership",
            snippet="",
        ) is True

    def test_empty_entity_always_returns_true(self):
        """When no primary entity exists, the filter must pass everything through."""
        from utils.query_builder import is_entity_relevant
        assert is_entity_relevant(
            entity="",
            title="OpenAI launches GPT-5",
            snippet="Unrelated article text.",
        ) is True, "Empty entity must be a passthrough (always True)"

    def test_partial_entity_name_match(self):
        """'Anthropic' should match 'Anthropic AI' in snippet."""
        from utils.query_builder import is_entity_relevant
        assert is_entity_relevant(
            entity="Anthropic",
            title="Major AI investments",
            snippet="Anthropic AI secures additional funding from Google.",
        ) is True


# ---------------------------------------------------------------------------
# primary_entity persisted in state
# ---------------------------------------------------------------------------

class TestPrimaryEntityInState:
    """Verify url_discovery stores primary_entity when companies are found."""

    def _make_state(self):
        return {
            "original_query": "Hugging Face competitors",
            "company_domains": [],
            "trusted_domains": [],
            "primary_entity": "",
            "guardrail_status": "valid",
            "guardrail_reason": "",
            "guardrail_blocked": False,
            "logs": [],
        }

    def test_primary_entity_stored_when_companies_found(self, monkeypatch):
        import utils.geturl as gu

        # Monkeypatch heavy async functions to avoid LLM/Tavily calls
        async def _fake_extract_companies(query):
            return ["Hugging Face"]

        async def _fake_find_official_domains(companies):
            return ["huggingface.co"]

        async def _fake_detect_category(query):
            return "ai"

        monkeypatch.setattr(gu, "extract_companies", _fake_extract_companies)
        monkeypatch.setattr(gu, "find_official_domains", _fake_find_official_domains)
        monkeypatch.setattr(gu, "detect_category", _fake_detect_category)
        monkeypatch.setattr(gu, "get_domains_by_category", lambda cat: ["techcrunch.com"])

        import asyncio
        state = self._make_state()
        result = asyncio.run(gu.url_discovery(state))

        assert result["primary_entity"] == "Hugging Face", (
            "url_discovery must store the first extracted company as primary_entity"
        )

    def test_primary_entity_empty_when_no_companies(self, monkeypatch):
        import utils.geturl as gu

        async def _fake_extract_companies(query):
            return []

        async def _fake_find_official_domains(companies):
            return []

        async def _fake_detect_category(query):
            return "general"

        monkeypatch.setattr(gu, "extract_companies", _fake_extract_companies)
        monkeypatch.setattr(gu, "find_official_domains", _fake_find_official_domains)
        monkeypatch.setattr(gu, "detect_category", _fake_detect_category)
        monkeypatch.setattr(gu, "get_domains_by_category", lambda cat: [])

        import asyncio
        state = self._make_state()
        result = asyncio.run(gu.url_discovery(state))

        assert result["primary_entity"] == "", (
            "primary_entity must be empty string when no companies found"
        )


# ---------------------------------------------------------------------------
# Entity filter in multi_search_agent
# ---------------------------------------------------------------------------

class TestEntityFilterInMultiSearchAgent:
    """Verify the post-retrieval entity filter removes irrelevant trusted articles."""

    def _make_article(self, title: str, snippet: str = "", source_type: str = "trusted"):
        from models.schemas import Article
        return Article(
            title=title,
            url=f"https://example.com/{title[:10].replace(' ', '')}",
            snippet=snippet,
            published_date="2026-03-01",
            source_type=source_type,
            domain="techcrunch.com",
        )

    def test_entity_filter_removes_irrelevant_articles(self, monkeypatch):
        """Articles without the entity name in title+snippet must be removed."""
        import agents.multi_search_agent as msa

        # Build state with primary_entity set
        state = {
            "primary_entity": "Hugging Face",
        }

        relevant = self._make_article("Hugging Face raises $100M", "Hugging Face platform grows")
        irrelevant = self._make_article("OpenAI launches GPT-5", "Benchmark results for OpenAI")

        all_trusted = [relevant, irrelevant]

        from utils.query_builder import is_entity_relevant
        entity = "Hugging Face"
        filtered = [
            a for a in all_trusted
            if is_entity_relevant(entity, a.title or "", a.snippet or "")
        ]

        assert len(filtered) == 1
        assert filtered[0].title == "Hugging Face raises $100M"

    def test_entity_filter_passthrough_when_no_entity(self):
        """When primary_entity is empty, no articles should be removed."""
        from models.schemas import Article
        from utils.query_builder import is_entity_relevant

        articles = [
            Article(title="OpenAI news", url="https://a.com", snippet="...",
                    published_date="2026-03-01", source_type="trusted", domain="t.com"),
            Article(title="Google AI news", url="https://b.com", snippet="...",
                    published_date="2026-03-01", source_type="trusted", domain="t.com"),
        ]

        entity = ""
        filtered = [a for a in articles if is_entity_relevant(entity, a.title or "", a.snippet or "")]
        assert len(filtered) == 2, "Empty entity must pass all articles through"
