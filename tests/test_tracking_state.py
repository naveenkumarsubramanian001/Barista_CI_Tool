"""
Unit tests for Track Companies pipeline state initialization.
Verifies that the tracking and report generation states include
primary_entity and the necessary guardrail fields for the pipeline
to function correctly with entity-anchored search.
"""

import sys
import os
import pytest
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Mock required environment variables before local imports
os.environ["GROQ_API_KEY"] = "test"
os.environ["TAVILY_API_KEY"] = "test"
os.environ["SERPER_API_KEY"] = "test"


class TestCompanyTrackingState:
    
    def test_build_company_queries_entity_anchored(self):
        """Verify that track queries naturally include the company name."""
        from services.company_tracking import build_company_queries
        
        queries = build_company_queries("Anthropic")
        assert len(queries) > 0
        for q in queries:
            assert "Anthropic" in q, "Tracker query must include company name"

    def test_run_company_tracking_scan_state(self, monkeypatch):
        """
        Verify that run_company_tracking_scan initializes ResearchState
        with primary_entity and guardrail fields correctly.
        """
        import services.company_tracking as ct
        
        # Monkeypatch database and external calls so we can intercept the state
        captured_states = []
        
        def fake_get_company(company_id):
            return {"id": company_id, "name": "Anthropic", "url": "anthropic.com"}
            
        async def fake_find_official_domains(names):
            return ["anthropic.com"]
            
        async def fake_run_node(fn, state):
            # Capture state as it hits the first node and return it unchanged
            captured_states.append(state)
            return state

        monkeypatch.setattr(ct, "get_company", fake_get_company)
        monkeypatch.setattr(ct, "find_official_domains", fake_find_official_domains)
        monkeypatch.setattr(ct, "_run_node", fake_run_node)
        monkeypatch.setattr(ct, "update_company_scan_telemetry", lambda *a, **kw: None)
        monkeypatch.setattr(ct, "update_company_scan_timestamps", lambda *a, **kw: None)
        monkeypatch.setattr(ct, "add_company_update", lambda *a, **kw: False)

        result = asyncio.run(ct.run_company_tracking_scan(
            company_id=1,
            search_days=30,
            trigger="test",
            create_notifications=False
        ))
        
        assert len(captured_states) > 0, "No state was captured"
        state = captured_states[0]
        
        # Verify entity anchoring fields
        assert state.get("primary_entity") == "Anthropic"
        assert "venturebeat.com" in state.get("trusted_domains", [])
        
        # Verify guardrail bypassing fields
        assert state.get("guardrail_status") == "valid"
        assert state.get("guardrail_blocked") is False
        assert "report_guardrail" in state.get("retry_counts", {})
        assert state["retry_counts"]["report_guardrail"] == 0

    def test_generate_company_report_state(self, monkeypatch):
        """
        Verify that generate_company_report initializes ResearchState
        with primary_entity and guardrail fields correctly.
        """
        import services.company_tracking as ct
        
        captured_states = []
        
        def fake_get_company(company_id):
            return {"id": company_id, "name": "Anthropic"}
            
        def fake_get_company_updates_by_ids(cid, uids):
            return [{"url": "https://example.com/insight", "source_type": "trusted"}]
            
        async def fake_run_node(fn, state):
            captured_states.append(state)
            # Need to provide a final report to bypass the exception check
            state["final_report"] = {"executive_summary": "Test", "sections": [], "references": []}
            return state

        monkeypatch.setattr(ct, "get_company", fake_get_company)
        monkeypatch.setattr(ct, "get_company_updates_by_ids", fake_get_company_updates_by_ids)
        monkeypatch.setattr(ct, "_run_node", fake_run_node)
        monkeypatch.setattr(ct, "add_report_event", lambda *a, **kw: None)
        monkeypatch.setattr(ct, "add_notification", lambda *a, **kw: None)
        
        # Override file I/O operations
        import json
        monkeypatch.setattr(json, "dump", lambda *a, **kw: None)
        monkeypatch.setattr("builtins.open", lambda *a, **kw: monkeypatch)  # Mock context manager
        
        class MockOpen:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            
        monkeypatch.setattr("builtins.open", lambda *a, **kw: MockOpen())
        monkeypatch.setattr(ct, "generate_pdf", lambda *a, **kw: None)

        result = asyncio.run(ct.generate_company_report(company_id=1, update_ids=[1]))
        
        assert len(captured_states) > 0
        state = captured_states[0]
        
        assert state.get("primary_entity") == "Anthropic"
        assert state.get("guardrail_status") == "valid"
        assert state.get("guardrail_blocked") is False
        assert "report_guardrail" in state.get("retry_counts", {})
