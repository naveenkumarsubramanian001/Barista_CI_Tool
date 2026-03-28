import json
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest
from fastapi.testclient import TestClient


class DummyState:
    def __init__(self, values=None, next_tasks=None):
        self.values = values or {}
        self.next = next_tasks or []


class FakeGraphApp:
    def __init__(self):
        self._states = {}

    async def ainvoke(self, initial_state, config=None):
        thread_id = config["configurable"]["thread_id"]
        if initial_state is not None:
            self._states[thread_id] = DummyState(
                values={
                    **initial_state,
                    "final_ranked_output": {
                        "official_sources": [
                            {
                                "title": "Official A",
                                "url": "https://official.example/a",
                                "snippet": "official snippet",
                                "published_date": "2026-03-01",
                                "source_type": "official",
                                "domain": "official.example",
                                "score": 0.88,
                            }
                        ],
                        "trusted_sources": [
                            {
                                "title": "Trusted A",
                                "url": "https://trusted.example/a",
                                "snippet": "trusted snippet",
                                "published_date": "2026-03-02",
                                "source_type": "trusted",
                                "domain": "trusted.example",
                                "score": 0.81,
                            }
                        ],
                    },
                    "logs": ["search done", "rank done"],
                },
                next_tasks=["summariser"],
            )
            return self._states[thread_id].values

        # Resume call after selection.
        state = self._states[thread_id]
        state.values["final_report"] = {
            "report_title": "Test Report",
            "executive_summary": "Summary",
            "conflict_and_consensus": "Consensus",
            "official_insights": [{"title": "O", "detailed_summary": "d", "reasoning": "r", "sentiment": "Neutral", "key_metrics": [], "key_features": [], "citation_id": 1}],
            "trusted_insights": [{"title": "T", "detailed_summary": "d", "reasoning": "r", "sentiment": "Neutral", "key_metrics": [], "key_features": [], "citation_id": 2}],
            "references": [],
        }
        state.next = []
        return state.values

    def get_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        return self._states.get(thread_id)

    def update_state(self, config, patch, as_node=None):
        thread_id = config["configurable"]["thread_id"]
        state = self._states.setdefault(thread_id, DummyState(values={}, next_tasks=["summariser"]))
        state.values.update(patch)


class FakeAnalyzerApp:
    def __init__(self):
        self._states = {}

    async def ainvoke(self, initial_state, config=None):
        thread_id = config["configurable"]["thread_id"]
        values = {
            **initial_state,
            "workflow_status": "completed",
            "progress_percentage": 100,
            "final_report": {
                "report_title": "Analyzer Report",
                "executive_summary": "Analyzer summary",
                "competitors": [],
                "user_product_positioning": "Position",
                "recommendations": ["Do X"],
            },
        }
        self._states[thread_id] = DummyState(values=values, next_tasks=[])
        return values

    def get_state(self, config):
        thread_id = config["configurable"]["thread_id"]
        return self._states.get(thread_id)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    # Stub heavy runtime modules before importing api.py.
    langgraph_mod = types.ModuleType("langgraph")
    checkpoint_mod = types.ModuleType("langgraph.checkpoint")
    memory_mod = types.ModuleType("langgraph.checkpoint.memory")

    class MemorySaver:
        pass

    memory_mod.MemorySaver = MemorySaver
    sys.modules["langgraph"] = langgraph_mod
    sys.modules["langgraph.checkpoint"] = checkpoint_mod
    sys.modules["langgraph.checkpoint.memory"] = memory_mod

    graph_pkg = types.ModuleType("graph")
    workflow_mod = types.ModuleType("graph.workflow")
    analyzer_workflow_mod = types.ModuleType("graph.analyzer_workflow")
    workflow_mod.build_graph = lambda checkpointer=None: None
    analyzer_workflow_mod.build_analyzer_graph = lambda checkpointer=None: None
    sys.modules["graph"] = graph_pkg
    sys.modules["graph.workflow"] = workflow_mod
    sys.modules["graph.analyzer_workflow"] = analyzer_workflow_mod

    database_mod = types.ModuleType("database")
    database_mod.create_db_and_tables = lambda: None
    sys.modules["database"] = database_mod

    scheduler_mod = types.ModuleType("scheduler")
    scheduler_mod.start_scheduler = lambda: None
    sys.modules["scheduler"] = scheduler_mod

    dotenv_mod = types.ModuleType("dotenv")
    dotenv_mod.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_mod

    ollama_mod = types.ModuleType("langchain_ollama")

    class _DummyEmbeddings:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyChat:
        def __init__(self, *args, **kwargs):
            pass

    ollama_mod.OllamaEmbeddings = _DummyEmbeddings
    ollama_mod.ChatOllama = _DummyChat
    sys.modules["langchain_ollama"] = ollama_mod

    from fastapi import APIRouter, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse
    import secrets
    import os
    import asyncio

    routers_pkg = types.ModuleType("routers")
    companies_mod = types.ModuleType("routers.companies")
    analyze_mod = types.ModuleType("routers.analyze")
    companies_mod.router = APIRouter(prefix="/api/companies", tags=["companies"])
    analyze_router = APIRouter(prefix="/api/analyze", tags=["analyze"])

    @analyze_router.post("/upload")
    async def upload_document(file: UploadFile = File(...)):
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="No readable text found in document.")
        session_id = f"analyzer_{secrets.token_hex(8)}"
        from api import analyzer_app, get_config

        initial_state = {
            "session_id": session_id,
            "uploaded_text": content.decode("utf-8", errors="ignore"),
            "product_profile": None,
            "discovered_competitors": [],
            "competitor_data": {},
            "final_report": None,
            "logs": ["start"],
            "workflow_status": "extracting",
            "progress_percentage": 10,
            "error": None,
        }

        async def run_analysis():
            final_output = await analyzer_app.ainvoke(initial_state, config=get_config(session_id))
            if final_output and final_output.get("final_report"):
                Path(f"analyze_report_{session_id}.pdf").write_bytes(b"%PDF-1.4\n%fake comparative\n")

        asyncio.create_task(run_analysis())
        return {"session_id": session_id, "status": "started", "message": "Document uploaded successfully"}

    @analyze_router.get("/status/{session_id}")
    async def status_document(session_id: str):
        from api import analyzer_app, get_config

        state = analyzer_app.get_state(get_config(session_id))
        if not state or not state.values:
            return {"session_id": session_id, "status": "initializing", "progress_percentage": 0, "logs": []}

        has_report = state.values.get("final_report") is not None
        pdf_exists = os.path.exists(f"analyze_report_{session_id}.pdf")
        status = "completed" if has_report and pdf_exists else state.values.get("workflow_status", "running")
        progress = 100 if has_report and pdf_exists else state.values.get("progress_percentage", 10)
        return {
            "session_id": session_id,
            "status": status,
            "progress_percentage": progress,
            "logs": state.values.get("logs", []),
            "report_data": state.values.get("final_report") if has_report else None,
        }

    @analyze_router.get("/download/{session_id}")
    async def download_document(session_id: str):
        file_path = f"analyze_report_{session_id}.pdf"
        if os.path.exists(file_path):
            return FileResponse(path=file_path, filename="Comparative_Intelligence_Report.pdf", media_type="application/pdf")
        raise HTTPException(status_code=404, detail="PDF not found on disk.")

    analyze_mod.router = analyze_router
    routers_pkg.companies = companies_mod
    routers_pkg.analyze = analyze_mod
    sys.modules["routers"] = routers_pkg
    sys.modules["routers.companies"] = companies_mod
    sys.modules["routers.analyze"] = analyze_mod

    api = importlib.import_module("api")

    monkeypatch.chdir(tmp_path)

    fake_graph = FakeGraphApp()
    fake_analyzer = FakeAnalyzerApp()

    monkeypatch.setattr(api, "graph_app", fake_graph)
    monkeypatch.setattr(api, "analyzer_app", fake_analyzer)
    monkeypatch.setattr(api, "create_db_and_tables", lambda: None)
    monkeypatch.setattr(api, "start_scheduler", lambda: None)

    def fake_create_task(coro):
        # Avoid pending background tasks during tests.
        coro.close()
        return SimpleNamespace(cancel=lambda: None)

    monkeypatch.setattr(api.asyncio, "create_task", fake_create_task)

    pdf_stub = types.ModuleType("utils.pdf_report")
    comp_pdf_stub = types.ModuleType("utils.comparative_pdf_report")

    def fake_generate_pdf(json_path, pdf_path):
        Path(pdf_path).write_bytes(b"%PDF-1.4\n%fake\n")

    def fake_generate_comparative_pdf(json_path, pdf_path):
        Path(pdf_path).write_bytes(b"%PDF-1.4\n%fake comparative\n")

    pdf_stub.generate_pdf = fake_generate_pdf
    comp_pdf_stub.generate_comparative_pdf = fake_generate_comparative_pdf
    sys.modules["utils.pdf_report"] = pdf_stub
    sys.modules["utils.comparative_pdf_report"] = comp_pdf_stub

    with TestClient(api.app) as test_client:
        yield test_client, fake_graph, fake_analyzer, api


def test_search_start_contract(client):
    test_client, _, _, _ = client
    response = test_client.post("/api/search", json={"query": "openai roadmap"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "started"
    assert payload["session_id"].startswith("sess_")


def test_workflow_pause_and_resume_article_selection_and_report_fetch(client):
    test_client, fake_graph, _, _ = client

    start_resp = test_client.post("/api/search", json={"query": "edge ai trends"}).json()
    session_id = start_resp["session_id"]

    # Inject paused state after phase-1.
    fake_graph._states[session_id] = DummyState(
        values={
            "original_query": "edge ai trends",
            "logs": ["phase1 done"],
            "final_ranked_output": {
                "official_sources": [
                    {
                        "title": "Official A",
                        "url": "https://official.example/a",
                        "snippet": "official snippet",
                        "published_date": "2026-03-01",
                        "source_type": "official",
                        "domain": "official.example",
                        "score": 0.88,
                    }
                ],
                "trusted_sources": [
                    {
                        "title": "Trusted A",
                        "url": "https://trusted.example/a",
                        "snippet": "trusted snippet",
                        "published_date": "2026-03-02",
                        "source_type": "trusted",
                        "domain": "trusted.example",
                        "score": 0.81,
                    }
                ],
            },
            "final_report": None,
        },
        next_tasks=["summariser"],
    )

    status_resp = test_client.get(f"/api/workflow/status/{session_id}")
    assert status_resp.status_code == 200
    status_payload = status_resp.json()
    assert status_payload["status"] == "completed"
    assert status_payload["current_stage"] == "select"

    articles_resp = test_client.get(f"/api/articles/{session_id}")
    assert articles_resp.status_code == 200
    articles_payload = articles_resp.json()
    assert articles_payload["total_count"] == 2

    generate_resp = test_client.post(
        f"/api/generate-pdf/{session_id}",
        json={"selected_article_urls": ["https://official.example/a", "https://trusted.example/a"]},
    )
    assert generate_resp.status_code == 200
    gen_payload = generate_resp.json()
    assert gen_payload["status"] == "success"

    report_resp = test_client.get(f"/api/report/{session_id}")
    assert report_resp.status_code == 200
    report_payload = report_resp.json()
    assert report_payload["session_id"] == session_id
    assert report_payload["report"]["report_title"] == "Test Report"

    pdf_status_resp = test_client.get(f"/api/pdf-status/{session_id}")
    assert pdf_status_resp.status_code == 200
    assert pdf_status_resp.json()["status"] == "completed"


def test_analyze_upload_status_download_contract(client):
    test_client, _, fake_analyzer, _ = client

    upload_resp = test_client.post(
        "/api/analyze/upload",
        files={"file": ("sample.txt", b"Product overview text", "text/plain")},
    )
    assert upload_resp.status_code == 200
    upload_payload = upload_resp.json()
    assert upload_payload["status"] == "started"
    assert upload_payload["session_id"].startswith("analyzer_")

    session_id = upload_payload["session_id"]

    # Set analyzer state and create expected pdf artifact.
    fake_analyzer._states[session_id] = DummyState(
        values={
            "workflow_status": "completed",
            "progress_percentage": 100,
            "logs": ["done"],
            "final_report": {"report_title": "Analyzer Report"},
            "error": None,
        },
        next_tasks=[],
    )
    Path(f"analyze_report_{session_id}.pdf").write_bytes(b"%PDF-1.4\n%analyzer\n")

    status_resp = test_client.get(f"/api/analyze/status/{session_id}")
    assert status_resp.status_code == 200
    status_payload = status_resp.json()
    assert status_payload["status"] == "completed"
    assert status_payload["progress_percentage"] == 100

    download_resp = test_client.get(f"/api/analyze/download/{session_id}")
    assert download_resp.status_code == 200
    assert download_resp.headers["content-type"].startswith("application/pdf")
