from pathlib import Path

import pytest


@pytest.mark.integration
def test_real_sqlite_checkpointer_pause_resume(tmp_path):
    pytest.importorskip("langgraph")
    pytest.importorskip("langgraph.checkpoint.sqlite")

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph
    from typing_extensions import TypedDict

    class State(TypedDict):
        value: int

    def step1(state: State):
        return {"value": state["value"] + 1}

    def step2(state: State):
        return {"value": state["value"] * 2}

    db_path = tmp_path / "checkpoints.sqlite"

    with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        wf = StateGraph(State)
        wf.add_node("step1", step1)
        wf.add_node("step2", step2)
        wf.add_edge(START, "step1")
        wf.add_edge("step1", "step2")
        wf.add_edge("step2", END)
        app = wf.compile(checkpointer=checkpointer, interrupt_before=["step2"])

        cfg = {"configurable": {"thread_id": "int-live-1"}}
        asyncio.run(app.ainvoke({"value": 1}, config=cfg))
        paused = app.get_state(cfg)
        assert "step2" in paused.next
        assert paused.values["value"] == 2

        asyncio.run(app.ainvoke(None, config=cfg))
        done = app.get_state(cfg)
        assert done.values["value"] == 4


@pytest.mark.integration
def test_real_analyze_router_upload_status_download(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("langgraph")
    pytest.importorskip("dotenv")
    pytest.importorskip("langchain_ollama")
    pytest.importorskip("fitz")

    import sys

    for mod in ["api", "routers", "routers.analyze", "routers.companies"]:
        sys.modules.pop(mod, None)

    import api
    from fastapi.testclient import TestClient

    monkeypatch.chdir(tmp_path)

    class DummyState:
        def __init__(self, values=None):
            self.values = values or {}
            self.next = []

    class FakeAnalyzer:
        def __init__(self):
            self._states = {}

        async def ainvoke(self, initial_state, config=None):
            sid = config["configurable"]["thread_id"]
            values = {
                **initial_state,
                "workflow_status": "completed",
                "progress_percentage": 100,
                "final_report": {
                    "report_title": "Analyzer",
                    "executive_summary": "Summary",
                    "competitors": [],
                    "user_product_positioning": "Position",
                    "recommendations": ["Action"],
                },
                "error": None,
            }
            self._states[sid] = DummyState(values=values)
            return values

        def get_state(self, config):
            return self._states.get(config["configurable"]["thread_id"])

        def update_state(self, config, patch):
            sid = config["configurable"]["thread_id"]
            state = self._states.setdefault(sid, DummyState(values={}))
            state.values.update(patch)

    analyzer = FakeAnalyzer()
    api.app.state.analyzer_app = analyzer
    api.app.state.get_config = api.get_config

    with TestClient(api.app) as client:
        up = client.post(
            "/api/analyze/upload",
            files={"file": ("sample.txt", b"some product text", "text/plain")},
        )
        assert up.status_code == 200
        payload = up.json()
        sid = payload["session_id"]

        analyzer._states[sid] = DummyState(
            values={
                "workflow_status": "completed",
                "progress_percentage": 100,
                "logs": ["done"],
                "final_report": {"report_title": "Analyzer"},
                "error": None,
            }
        )
        Path(f"analyze_report_{sid}.pdf").write_bytes(b"%PDF-1.4\n%integration\n")

        status = client.get(f"/api/analyze/status/{sid}")
        assert status.status_code == 200
        status_payload = status.json()
        assert status_payload["status"] == "completed"

        dl = client.get(f"/api/analyze/download/{sid}")
        assert dl.status_code == 200
        assert dl.headers["content-type"].startswith("application/pdf")


@pytest.mark.integration
def test_real_comparative_pdf_integrity(tmp_path):
    pypdf = pytest.importorskip("pypdf")

    from utils.comparative_pdf_report import generate_comparative_pdf

    data = {
        "report_title": "Integration Report",
        "executive_summary": "Integration summary",
        "competitors": [
            {
                "name": "Comp One",
                "domain": "one.example",
                "strengths": ["Strong brand"],
                "weaknesses": ["High cost"],
                "pricing_strategy": "Premium",
                "key_features": ["Feature X"],
            }
        ],
        "user_product_positioning": "Affordable quality",
        "recommendations": ["Improve distribution"],
    }

    src = tmp_path / "in.json"
    out = tmp_path / "out.pdf"
    src.write_text(__import__("json").dumps(data), encoding="utf-8")

    generate_comparative_pdf(str(src), str(out))

    assert out.exists()
    reader = pypdf.PdfReader(str(out))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    assert "Executive Summary" in text
    assert "Comp One" in text
    assert "Recommendations" in text
