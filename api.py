"""
Main FastAPI application entry point for the Barista CI backend.
This file sets up the API routes, CORS middleware, checkpointer state management,
and orchestrates the asynchronous execution of the LangGraph workflows (`graph_app`, `analyzer_app`).
It handles REST endpoints for starting searches, getting workflow status, retrieving articles,
and generating final PDF reports.
"""
import asyncio
import os
import secrets
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from langgraph.checkpoint.memory import MemorySaver
from graph.workflow import build_graph
from graph.analyzer_workflow import build_analyzer_graph
from config import (
    CORS_ALLOWED_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CHECKPOINTER_BACKEND,
    CHECKPOINTER_SQLITE_PATH,
)

from contextlib import asynccontextmanager
from database import create_db_and_tables
from scheduler import start_scheduler, register_scan_handler
from services.company_tracking import run_company_tracking_scan

_checkpointer_cm = None
checkpointer = None
graph_app = None
analyzer_app = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer, graph_app, analyzer_app

    create_db_and_tables()
    register_scan_handler(run_company_tracking_scan)
    start_scheduler()

    existing_graph = getattr(app.state, "graph_app", None)
    existing_analyzer = getattr(app.state, "analyzer_app", None)
    if existing_graph is None or existing_analyzer is None:
        checkpointer = await _build_checkpointer()
        graph_app = build_graph(checkpointer=checkpointer)
        analyzer_app = build_analyzer_graph(checkpointer=checkpointer)
        app.state.graph_app = graph_app
        app.state.analyzer_app = analyzer_app
    else:
        graph_app = existing_graph
        analyzer_app = existing_analyzer

    try:
        yield
    finally:
        global _checkpointer_cm
        if _checkpointer_cm is not None:
            await _checkpointer_cm.__aexit__(None, None, None)
            _checkpointer_cm = None

app = FastAPI(title="Barista CI API", lifespan=lifespan)

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _build_checkpointer():
    """Choose checkpointer backend from env, with safe fallback to in-memory."""
    if CHECKPOINTER_BACKEND == "sqlite":
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            candidate = AsyncSqliteSaver.from_conn_string(CHECKPOINTER_SQLITE_PATH)
            if hasattr(candidate, "__aenter__") and hasattr(candidate, "__aexit__"):
                global _checkpointer_cm
                _checkpointer_cm = candidate
                return await _checkpointer_cm.__aenter__()
            return candidate
        except Exception as exc:
            print(
                f"[startup] Could not initialize sqlite checkpointer ({exc}). "
                "Falling back to MemorySaver."
            )

    if CHECKPOINTER_BACKEND != "memory":
        print(
            f"[startup] Unsupported CHECKPOINTER_BACKEND='{CHECKPOINTER_BACKEND}'. "
            "Falling back to MemorySaver."
        )

    return MemorySaver()

from routers import companies, analyze
app.include_router(companies.router)
app.include_router(analyze.router)


def _default_stages():
    return [
        {"id": "understand", "label": "Understanding your query", "status": "pending"},
        {"id": "identify", "label": "Identifying topic & domain", "status": "pending"},
        {"id": "collect", "label": "Collecting source candidates", "status": "pending"},
        {"id": "filter", "label": "Filtering & ranking content", "status": "pending"},
        {"id": "analyze", "label": "Analyzing selected documents", "status": "pending"},
        {"id": "prepare", "label": "Preparing results for review", "status": "pending"},
    ]


# --- Pydantic Models for API ---
class SearchRequest(BaseModel):
    query: str


class GeneratePdfRequest(BaseModel):
    selected_article_urls: List[str]


# --- Helper functions ---
def get_config(session_id: str):
    return {"configurable": {"thread_id": session_id}}


async def _get_graph_state(graph, config):
    """Support both async LangGraph APIs and sync test doubles."""
    if hasattr(graph, "aget_state"):
        return await graph.aget_state(config)
    return graph.get_state(config)


async def _update_graph_state(graph, config, values):
    """Support both async LangGraph APIs and sync test doubles."""
    if hasattr(graph, "aupdate_state"):
        return await graph.aupdate_state(config, values)
    return graph.update_state(config, values)


async def _ainvoke_graph(graph, input_value, config, **kwargs):
    """Invoke graph with optional advanced kwargs, with test-double compatibility."""
    if hasattr(graph, "ainvoke"):
        try:
            return await graph.ainvoke(input_value, config=config, **kwargs)
        except TypeError:
            return await graph.ainvoke(input_value, config=config)
    return graph.invoke(input_value, config=config)


def _map_article(a, category: str, session_id: str, idx: int, total_per_category: int):
    """Map an article object or dict to the API response format."""
    # Mark top 3 articles in each category as default_selected
    default_selected = idx < 3
    score_raw = a.get("score") if isinstance(a, dict) else getattr(a, "score", None)
    # Normalise score to 0-1 range for frontend (frontend multiplies by 100)
    if score_raw is None:
        score = 1.0
    elif isinstance(score_raw, int) and score_raw > 1:
        score = round(score_raw / 100.0, 2)
    else:
        score = float(score_raw)

    return {
        "id": a.get("url") if isinstance(a, dict) else a.url,
        "session_id": session_id,
        "title": a.get("title") if isinstance(a, dict) else a.title,
        "url": a.get("url") if isinstance(a, dict) else a.url,
        "domain": (a.get("domain") or "unknown") if isinstance(a, dict) else (getattr(a, "domain", None) or "unknown"),
        "category": category,
        "score": score,
        "snippet": a.get("snippet") if isinstance(a, dict) else getattr(a, "snippet", None),
        "is_approved": True,
        "default_selected": default_selected,
        "user_selected": False,
        "published_date": a.get("published_date") if isinstance(a, dict) else getattr(a, "published_date", None),
    }


# --- Endpoints ---


@app.get("/api/healthz")
async def health_check():
    return {"status": "ok"}


@app.get("/api/tips")
async def get_tips():
    return {
        "tips": [
            {
                "id": "1",
                "text": "What is OpenAI's latest product?",
                "category": "product",
            },
            {"id": "2", "text": "Recent advancements in Edge AI", "category": "domain"},
            {
                "id": "3",
                "text": "Apple's 2026 hardware roadmap details",
                "category": "company",
            },
            {
                "id": "4",
                "text": "Anthropic vs OpenAI competitor analysis",
                "category": "company",
            },
        ]
    }


@app.post("/api/search")
async def start_search(request: SearchRequest):
    session_id = f"sess_{secrets.token_hex(8)}"

    initial_state = {
        "original_query": request.query,
        "subqueries": [],
        "official_sources": [],
        "trusted_sources": [],
        "final_ranked_output": {},
        "final_report": None,
        "company_domains": [],
        "trusted_domains": [],
        "validation_feedback": "",
        "validation_passed": False,
        "validation_metrics": {},
        "decomposition_score": 0.0,
        "redundancy_pairs": [],
        "coverage_gaps": [],
        "semantic_warnings": [],
        "retry_counts": {"decomposer": 0, "search": 0, "summariser": 0, "report_guardrail": 0},
        "error": None,
        "search_days_used": None,
        "selected_articles": [],
        "logs": [],
        "stages": _default_stages(),
        "current_stage": "understand",
        "progress_percentage": 5,
        # Guardrail AI — initialised to safe defaults
        "guardrail_status": "unchecked",
        "guardrail_reason": "",
        "guardrail_blocked": False,
        # Entity anchoring — populated by url_discovery node
        "primary_entity": "",
    }

    async def run_phase_1():
        print(
            f"[{session_id}] 🚀 Starting LangGraph Workflow Phase 1 (Search & Filter)..."
        )
        await _ainvoke_graph(
            graph_app,
            initial_state,
            config=get_config(session_id),
            interrupt_before=["summariser"],
        )
        print(
            f"[{session_id}] ⏳ Phase 1 paused. Waiting for human-in-the-loop selection..."
        )

    asyncio.create_task(run_phase_1())

    return {"session_id": session_id, "status": "started"}


@app.get("/api/workflow/status/{session_id}")
async def get_workflow_status(session_id: str):
    config = get_config(session_id)
    state = await _get_graph_state(graph_app, config)

    if not state or not state.values:
        stages = _default_stages()
        stages[0]["status"] = "running"
        return {
            "session_id": session_id,
            "status": "pending",
            "current_stage": "initializing",
            "progress_percentage": 0,
            "stages": stages,
            "logs": [],
        }

    next_tasks = state.next
    is_paused_for_human = "summariser" in next_tasks
    has_final_report = bool(state.values.get("final_report"))
    is_blocked = bool(state.values.get("guardrail_blocked"))

    logs = state.values.get("logs", [])
    stages = state.values.get("stages", _default_stages())
    progress = int(state.values.get("progress_percentage", 0) or 0)
    current_stage = state.values.get("current_stage", "searching")

    # Determine overall status and progress
    if is_blocked:
        # Query was blocked by the guardrail — surface this cleanly
        return {
            "session_id": session_id,
            "status": "blocked",
            "current_stage": "guardrail",
            "progress_percentage": 0,
            "stages": _default_stages(),
            "logs": logs,
            "guardrail_reason": state.values.get("guardrail_reason", "Query blocked by security guardrail."),
            "message": "Your query was blocked: " + state.values.get(
                "guardrail_reason",
                "This query is outside the scope of the system or violates security guidelines."
            ),
        }
    elif has_final_report:
        status = "completed"
        progress = 100
        current_stage = "finished"
        stages = [
            {"id": s["id"], "label": s["label"], "status": "completed"}
            for s in _default_stages()
        ]
    elif is_paused_for_human:
        status = "completed"
        progress = max(progress, 100)
        current_stage = "select"
        stages = [
            {
                "id": stage.get("id"),
                "label": stage.get("label"),
                "status": "completed" if stage.get("status") != "failed" else "failed",
            }
            for stage in stages
        ]
    else:
        status = "running"
        if progress <= 0:
            progress = 10

    return {
        "session_id": session_id,
        "status": status,
        "current_stage": current_stage,
        "progress_percentage": progress,
        "stages": stages,
        "logs": logs,
    }


@app.get("/api/articles/{session_id}")
async def get_scored_articles(session_id: str):
    config = get_config(session_id)
    state = await _get_graph_state(graph_app, config)

    if not state or not state.values:
        raise HTTPException(
            status_code=404, detail="Session state not found or not initialized."
        )

    final_ranked = state.values.get("final_ranked_output", {})
    official_top = final_ranked.get("official_sources", [])
    trusted_top = final_ranked.get("trusted_sources", [])

    off_mapped = [
        _map_article(a, "official", session_id, i, len(official_top))
        for i, a in enumerate(official_top)
    ]
    tr_mapped = [
        _map_article(a, "trusted", session_id, i, len(trusted_top))
        for i, a in enumerate(trusted_top)
    ]

    return {
        "session_id": session_id,
        "official_articles": off_mapped,
        "trusted_articles": tr_mapped,
        "total_count": len(off_mapped) + len(tr_mapped),
    }


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    config = get_config(session_id)
    state = await _get_graph_state(graph_app, config)
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "query": state.values.get("original_query", "Research Session"),
        "workflow_status": "active",
    }


@app.get("/api/report/{session_id}")
async def get_report(session_id: str):
    """Return the final_report JSON for inline rendering in the frontend."""
    config = get_config(session_id)
    state = await _get_graph_state(graph_app, config)

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Session not found")

    final_report = state.values.get("final_report")
    if not final_report:
        # Try loading from disk if it was saved
        json_path = f"report_{session_id}.json"
        if os.path.exists(json_path):
            import json

            with open(json_path, "r", encoding="utf-8") as f:
                final_report = json.load(f)
        else:
            raise HTTPException(status_code=404, detail="Report not yet generated.")

    # Ensure a downloadable PDF exists for completed sessions.
    report_pdf_path = f"report_{session_id}.pdf"
    if final_report and not os.path.exists(report_pdf_path):
        try:
            import json
            from utils.pdf_report import generate_pdf as gen_pdf_file

            report_json_name = f"report_{session_id}.json"
            with open(report_json_name, "w", encoding="utf-8") as f:
                json.dump(final_report, f, indent=2)
            gen_pdf_file(report_json_name, report_pdf_path)
        except Exception as exc:
            print(f"[{session_id}] Could not generate missing PDF in /api/report: {exc}")

    return {
        "session_id": session_id,
        "report": final_report,
        "pdf_url": f"/api/pdf/download/{session_id}"
        if os.path.exists(report_pdf_path)
        else None,
    }


@app.post("/api/generate-pdf/{session_id}")
async def generate_pdf(session_id: str, request: GeneratePdfRequest):
    config = get_config(session_id)
    state = await _get_graph_state(graph_app, config)

    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Idempotent behavior: if report is already generated, return/download it
    # instead of failing with a workflow-state error.
    existing_report = state.values.get("final_report") if state.values else None
    existing_pdf = f"report_{session_id}.pdf"
    if existing_report:
        if not os.path.exists(existing_pdf):
            import json
            from utils.pdf_report import generate_pdf as gen_pdf_file

            report_json_name = f"report_{session_id}.json"
            with open(report_json_name, "w", encoding="utf-8") as f:
                json.dump(existing_report, f, indent=2)
            gen_pdf_file(report_json_name, existing_pdf)

        return {
            "session_id": session_id,
            "status": "success",
            "message": "Report already generated.",
            "pdf_url": f"/api/pdf/download/{session_id}",
        }

    if not state.next or "summariser" not in state.next:
        raise HTTPException(
            status_code=400,
            detail="Workflow is not paused at article-selection stage yet. Wait for source review to finish, then retry.",
        )

    if not request.selected_article_urls:
        raise HTTPException(
            status_code=400,
            detail="Please select at least one insight before generating the final report.",
        )

    print(
        f"[{session_id}] Received user selection. {len(request.selected_article_urls)} articles selected."
    )

    # Update state with the selected URLs
    await _update_graph_state(
        graph_app, config, {"selected_articles": request.selected_article_urls}
    )

    print(f"[{session_id}] 🚀 Starting Phase 2 (Summarise & PDF)...")
    final_output = await _ainvoke_graph(
        graph_app,
        None,
        config=config,
        interrupt_before=[],
    )

    if final_output.get("final_report"):
        import json

        report_json_name = f"report_{session_id}.json"
        report_pdf_name = f"report_{session_id}.pdf"

        with open(report_json_name, "w") as f:
            json.dump(final_output["final_report"], f, indent=2)

        from utils.pdf_report import generate_pdf as gen_pdf_file

        gen_pdf_file(report_json_name, report_pdf_name)

        return {
            "session_id": session_id,
            "status": "success",
            "message": "Report generated successfully.",
            "pdf_url": f"/api/pdf/download/{session_id}",
        }
    else:
        raise HTTPException(
            status_code=500, detail="Failed to generate the report data."
        )


@app.get("/api/pdf-status/{session_id}")
async def get_pdf_status(session_id: str):
    config = get_config(session_id)
    state = await _get_graph_state(graph_app, config)

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Session not found")

    has_report = state.values.get("final_report") is not None
    pdf_exists = os.path.exists(f"report_{session_id}.pdf")

    if has_report and pdf_exists:
        return {
            "session_id": session_id,
            "status": "completed",
            "download_url": f"/api/pdf/download/{session_id}",
        }
    elif has_report:
        return {"session_id": session_id, "status": "generating"}

    next_tasks = state.next
    if next_tasks and "summariser" in next_tasks:
        return {"session_id": session_id, "status": "pending"}

    return {"session_id": session_id, "status": "generating"}


@app.get("/api/pdf/download/{session_id}")
async def download_pdf(session_id: str):
    file_path = f"report_{session_id}.pdf"
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename="Competitive_Intelligence_Report.pdf",
            media_type="application/pdf",
        )
    raise HTTPException(status_code=404, detail="PDF not found on disk.")


@app.get("/api/article-content")
async def fetch_article_content(url: str):
    """
    Proxy endpoint to extract article content for the in-app reader.
    Uses requests + BeautifulSoup to extract article text.
    Falls back gracefully with a clear error if extraction fails.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove nav, script, style, footer, ads
        for tag in soup(
            ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]
        ):
            tag.decompose()

        # Try article body first
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else "Article"

        # Try semantic article tag or main
        article = (
            soup.find("article")
            or soup.find("main")
            or soup.find(id=lambda x: x and "article" in x.lower() if x else False)
            or soup.find(
                class_=lambda x: x and "article" in " ".join(x).lower() if x else False
            )
        )

        if article:
            content = article.get_text(separator="\n", strip=True)
        else:
            # Fallback: gather all paragraph text
            paragraphs = soup.find_all("p")
            content = "\n\n".join(
                p.get_text(strip=True)
                for p in paragraphs
                if len(p.get_text(strip=True)) > 60
            )

        # Trim very long content
        if len(content) > 8000:
            content = (
                content[:8000]
                + "\n\n[Content truncated for display. View full article at original URL.]"
            )

        return {
            "url": url,
            "title": title,
            "content": content if content else None,
            "can_embed": True,
            "error": None,
        }

    except requests.exceptions.Timeout:
        return {
            "url": url,
            "title": None,
            "content": None,
            "can_embed": False,
            "error": "Request timed out.",
        }
    except requests.exceptions.ConnectionError:
        return {
            "url": url,
            "title": None,
            "content": None,
            "can_embed": False,
            "error": "Could not connect to the article site.",
        }
    except requests.exceptions.HTTPError as e:
        return {
            "url": url,
            "title": None,
            "content": None,
            "can_embed": False,
            "error": f"HTTP {e.response.status_code}: Access denied or article not available.",
        }
    except ImportError:
        return {
            "url": url,
            "title": None,
            "content": None,
            "can_embed": False,
            "error": "Content extraction library not installed (requests/beautifulsoup4).",
        }
    except Exception as e:
        return {
            "url": url,
            "title": None,
            "content": None,
            "can_embed": False,
            "error": str(e),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
