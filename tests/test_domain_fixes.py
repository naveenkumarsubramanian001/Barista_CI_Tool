"""
Unit tests for the two domain-fix behaviours introduced in utils/geturl.py:

  Fix 1 — Trusted-domain deduplication:
    Official company domains must not also appear in the trusted_domains list.

  Fix 2 — Primary entity accuracy:
    When a query names a single primary company ('Anthropic competitors'),
    only that company's domain should be discovered as 'official'. Competitor
    names must not bleed into company_domains.

These tests are designed to run without network access and without live LLM
calls. All external dependencies are monkey-patched.
"""

import asyncio
import sys
import os
import pytest

# Ensure project root is on sys.path so relative imports work when running
# pytest from the Barista_CI_Tool_copy directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1 — Deduplication helpers (pure-function tests, no mocking required)
# ─────────────────────────────────────────────────────────────────────────────

class TestTrustedDomainDedup:
    """
    The dedup logic inside url_discovery() is a simple set-difference.
    These tests validate that logic in isolation so we don't need to spin
    up the full LangGraph workflow.
    """

    def _apply_dedup(self, company_domains, trusted_domains):
        """Mirrors the dedup snippet added to url_discovery()."""
        official_set = set(company_domains)
        return [d for d in trusted_domains if d not in official_set]

    def test_removes_official_domain_from_trusted(self):
        """If openai.com is an official domain it must be removed from trusted."""
        company_domains = ["openai.com"]
        trusted_domains = ["openai.com", "venturebeat.com", "techcrunch.com"]

        result = self._apply_dedup(company_domains, trusted_domains)

        assert "openai.com" not in result, (
            "Official domain 'openai.com' should be stripped from trusted list"
        )
        assert "venturebeat.com" in result
        assert "techcrunch.com" in result
        assert len(result) == 2

    def test_removes_multiple_official_domains(self):
        """All official domains that overlap with trusted should be removed."""
        company_domains = ["anthropic.com", "openai.com"]
        trusted_domains = ["anthropic.com", "openai.com", "syncedreview.com"]

        result = self._apply_dedup(company_domains, trusted_domains)

        assert "anthropic.com" not in result
        assert "openai.com" not in result
        assert "syncedreview.com" in result
        assert len(result) == 1

    def test_no_dedup_when_no_overlap(self):
        """Trusted list is unchanged when there is no overlap with official."""
        company_domains = ["anthropic.com"]
        trusted_domains = ["venturebeat.com", "techcrunch.com", "syncedreview.com"]

        result = self._apply_dedup(company_domains, trusted_domains)

        assert result == trusted_domains, (
            "Trusted list should be unchanged when no overlap exists"
        )

    def test_empty_company_domains_leaves_trusted_intact(self):
        """If no official domains found, trusted list is unchanged."""
        company_domains = []
        trusted_domains = ["venturebeat.com", "wired.com"]

        result = self._apply_dedup(company_domains, trusted_domains)

        assert result == trusted_domains

    def test_empty_trusted_domains(self):
        """Nothing to strip from an empty trusted list."""
        company_domains = ["openai.com"]
        trusted_domains = []

        result = self._apply_dedup(company_domains, trusted_domains)

        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 — extract_primary_entity() function
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractPrimaryEntity:
    """
    Tests for the new extract_primary_entity() async function.
    The LLM chain is monkey-patched so tests run offline.
    """

    def _make_chain_mock(self, primary_company_value):
        """
        Returns an object that mimics `primary_entity_chain.ainvoke()`.
        primary_company_value can be a string or None.
        """
        class FakeChain:
            async def ainvoke(self, payload):
                return {"primary_company": primary_company_value}
        return FakeChain()

    def test_single_company_query_returns_name(self, monkeypatch):
        """'Anthropic competitors' → returns 'Anthropic'."""
        import utils.geturl as geturl_mod
        monkeypatch.setattr(
            geturl_mod, "primary_entity_chain",
            self._make_chain_mock("Anthropic")
        )

        result = asyncio.run(
            geturl_mod.extract_primary_entity("Anthropic competitors")
        )

        assert result == "Anthropic", (
            f"Expected 'Anthropic' but got {result!r}"
        )

    def test_generic_query_returns_none(self, monkeypatch):
        """'best AI companies 2025' → returns None (no single primary)."""
        import utils.geturl as geturl_mod
        monkeypatch.setattr(
            geturl_mod, "primary_entity_chain",
            self._make_chain_mock(None)
        )

        result = asyncio.run(
            geturl_mod.extract_primary_entity("best AI companies 2025")
        )

        assert result is None

    def test_llm_failure_returns_none(self, monkeypatch):
        """If the LLM chain raises, the function swallows the error and returns None."""
        import utils.geturl as geturl_mod

        class BrokenChain:
            async def ainvoke(self, payload):
                raise RuntimeError("LLM unavailable")

        monkeypatch.setattr(geturl_mod, "primary_entity_chain", BrokenChain())

        result = asyncio.run(
            geturl_mod.extract_primary_entity("OpenAI competitors")
        )

        assert result is None, (
            "LLM failure should gracefully return None, not raise"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2 — extract_companies() resolution order
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractCompaniesResolutionOrder:
    """
    Verifies that extract_companies() respects the new resolution order:
      1. primary entity (query-text only)
      2. NER entities (if provided)
      3. dynamic suggestion (web search + LLM)
    """

    def test_primary_entity_shortcircuits_dynamic_suggestion(self, monkeypatch):
        """
        When a primary entity is found, suggest_companies_dynamic must NOT be
        called. This is the key guarantee that prevents competitor names from
        leaking into the official-domain list.
        """
        import utils.geturl as geturl_mod

        # Primary entity returns 'Anthropic'
        async def fake_primary(q): return "Anthropic"
        monkeypatch.setattr(geturl_mod, "extract_primary_entity", fake_primary)

        # Batch validation just echoes back the input
        async def fake_validate(entities): return entities
        monkeypatch.setattr(geturl_mod, "validate_companies_batch", fake_validate)

        # Dynamic suggestion should NOT be called — raise if it is
        async def should_not_be_called(q):
            raise AssertionError("suggest_companies_dynamic was called but should not have been")
        monkeypatch.setattr(geturl_mod, "suggest_companies_dynamic", should_not_be_called)

        result = asyncio.run(
            geturl_mod.extract_companies("Anthropic competitors")
        )

        assert result == ["Anthropic"], f"Expected ['Anthropic'], got {result}"

    def test_falls_back_to_dynamic_when_no_primary(self, monkeypatch):
        """
        When no primary entity is found AND no NER entities are provided,
        suggest_companies_dynamic should run.
        """
        import utils.geturl as geturl_mod

        async def fake_primary(q): return None
        monkeypatch.setattr(geturl_mod, "extract_primary_entity", fake_primary)

        async def fake_dynamic(q): return ["VentureBeats AI", "TechStartup"]
        monkeypatch.setattr(geturl_mod, "suggest_companies_dynamic", fake_dynamic)

        async def fake_validate(entities): return entities
        monkeypatch.setattr(geturl_mod, "validate_companies_batch", fake_validate)

        result = asyncio.run(
            geturl_mod.extract_companies("best AI companies for enterprise")
        )

        assert "VentureBeats AI" in result or "TechStartup" in result, (
            "Dynamic suggestion results should be used when no primary entity found"
        )
