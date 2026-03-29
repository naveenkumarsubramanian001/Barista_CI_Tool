import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")

import api


class DummyGraph:
    def __init__(self):
        self._states = {}

    async def ainvoke(self, state, config=None):
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "default")
        current = self._states.get(thread_id, SimpleNamespace(values={}, next=[]))

        # Initial phase call
        if state is not None:
            current.values = {
                **getattr(current, "values", {}),
                **(state or {}),
                "logs": (state or {}).get("logs", []),
            }
            current.next = ["summariser"]
            self._states[thread_id] = current
            return current.values

        # Resume call after human selection
        selected = current.values.get("selected_articles", [])
        current.values["final_report"] = {
            "title": "Contract Report",
            "selected_count": len(selected),
        }
        current.next = []
        self._states[thread_id] = current
        return current.values

    def get_state(self, config):
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        return self._states.get(thread_id)

    def update_state(self, config, values):
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        current = self._states.get(thread_id, SimpleNamespace(values={}, next=[]))
        current.values = {**getattr(current, "values", {}), **values}
        self._states[thread_id] = current


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(api, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(api, "start_scheduler", lambda: None)

    main_graph = DummyGraph()
    analyzer_graph = DummyGraph()

    monkeypatch.setattr(api, "graph_app", main_graph)
    monkeypatch.setattr(api, "analyzer_app", analyzer_graph)
    api.app.state.graph_app = main_graph
    api.app.state.analyzer_app = analyzer_graph

    with TestClient(api.app) as test_client:
        yield test_client


def test_healthz_contract(client):
    response = client.get("/api/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_and_status_contract(client):
    start = client.post("/api/search", json={"query": "espresso machine competitors"})
    assert start.status_code == 200
    payload = start.json()
    assert "session_id" in payload
    assert payload["status"] == "started"

    status = client.get(f"/api/workflow/status/{payload['session_id']}")
    assert status.status_code == 200
    status_payload = status.json()
    assert status_payload["session_id"] == payload["session_id"]
    assert "status" in status_payload
    assert "progress_percentage" in status_payload


def test_analyzer_upload_and_status_contract(client):
    files = {"file": ("product.txt", b"Barista AI brewer with analytics", "text/plain")}
    upload = client.post("/api/analyze/upload", files=files)
    assert upload.status_code == 200

    session_id = upload.json()["session_id"]
    status = client.get(f"/api/analyze/status/{session_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["session_id"] == session_id
    assert "status" in body
    assert "progress_percentage" in body
