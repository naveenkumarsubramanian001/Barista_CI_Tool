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


@pytest.fixture()
def temp_company_db(tmp_path, monkeypatch):
    import database

    db_path = tmp_path / "company_test.sqlite"
    monkeypatch.setattr(database, "DB_PATH", str(db_path))
    database.create_db_and_tables()
    return database


def test_company_scrape_contract(client, temp_company_db, monkeypatch):
    import routers.companies as companies_router
    import services.company_tracking as company_tracking

    def _drop_task(coro):
        # Avoid background task leakage during tests.
        coro.close()
        return None

    async def _fake_scan(company_id: int, *, search_days: int, trigger: str, create_notifications: bool):
        return {
            "company_id": company_id,
            "company_name": "OpenAI",
            "trigger": trigger,
            "search_days": search_days,
            "total_ranked": 4,
            "new_insights": 2,
        }

    monkeypatch.setattr(companies_router.asyncio, "create_task", _drop_task)
    monkeypatch.setattr(company_tracking, "run_company_tracking_scan", _fake_scan)

    created = client.post(
        "/api/companies/",
        json={"name": "OpenAI", "url": "https://openai.com"},
    )
    assert created.status_code == 200
    company_id = created.json()["id"]

    scrape = client.post(f"/api/companies/{company_id}/scrape")
    assert scrape.status_code == 200
    payload = scrape.json()
    assert payload["status"] == "ok"
    assert payload["company_id"] == company_id
    assert payload["trigger"] == "manual"
    assert payload["new_insights"] == 2


def test_company_update_read_and_notifications_contract(client, temp_company_db):
    company = temp_company_db.add_company("Anthropic", "https://anthropic.com")
    company_id = int(company["id"])

    update = temp_company_db.add_company_update(
        company_id,
        {
            "title": "New launch",
            "url": "https://example.com/news-1",
            "snippet": "Launch details",
            "source_type": "trusted",
            "published_date": "2026-03-29",
            "is_read": False,
            "metadata": {},
        },
    )
    assert update is not None
    update_id = int(update["id"])

    notification = temp_company_db.add_notification(
        title="New intelligence found",
        message="1 new insight",
        company_id=company_id,
    )
    notification_id = int(notification["id"])

    mark_update = client.post(f"/api/companies/{company_id}/updates/{update_id}/read")
    assert mark_update.status_code == 200
    assert mark_update.json() == {"status": "ok"}

    company_after = client.get(f"/api/companies/{company_id}")
    assert company_after.status_code == 200
    assert company_after.json()["unread_count"] == 0

    notifications = client.get("/api/companies/notifications?limit=10")
    assert notifications.status_code == 200
    body = notifications.json()
    assert "notifications" in body
    assert any(n["id"] == notification_id for n in body["notifications"])

    mark_notification = client.post(f"/api/companies/notifications/{notification_id}/read")
    assert mark_notification.status_code == 200
    assert mark_notification.json() == {"status": "ok"}


def test_company_generate_report_contract(client, temp_company_db, monkeypatch):
    import services.company_tracking as company_tracking

    async def _fake_generate(company_id: int, update_ids: list[int]):
        return {
            "company_id": company_id,
            "session_id": "company_1_abc123",
            "report": {"report_title": "Mock Report"},
            "pdf_url": "/api/pdf/download/company_1_abc123",
        }

    monkeypatch.setattr(company_tracking, "generate_company_report", _fake_generate)

    company = temp_company_db.add_company("Perplexity", "https://perplexity.ai")
    company_id = int(company["id"])

    empty_selection = client.post(
        f"/api/companies/{company_id}/generate-report",
        json={"update_ids": []},
    )
    assert empty_selection.status_code == 400
    assert empty_selection.json()["detail"] == "Select at least one insight"

    generated = client.post(
        f"/api/companies/{company_id}/generate-report",
        json={"update_ids": [1, 2]},
    )
    assert generated.status_code == 200
    payload = generated.json()
    assert payload["status"] == "ok"
    assert payload["company_id"] == company_id
    assert payload["session_id"] == "company_1_abc123"
    assert payload["pdf_url"] == "/api/pdf/download/company_1_abc123"
