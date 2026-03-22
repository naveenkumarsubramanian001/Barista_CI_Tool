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

from contextlib import asynccontextmanager
from database import create_db_and_tables
from scheduler import start_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    start_scheduler()
    yield

app = FastAPI(title="Barista CI API", lifespan=lifespan)

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routers import companies
app.include_router(companies.router)

# Global Checkpointer (MemorySaver) to maintain state across HTTP requests
checkpointer = MemorySaver()

# Global Graph instantiation
graph_app = build_graph(checkpointer=checkpointer)


# --- Pydantic Models for API ---
class SearchRequest(BaseModel):
    query: str


class GeneratePdfRequest(BaseModel):
    selected_article_urls: List[str]


# --- Helper functions ---
def get_config(session_id: str):
    return {"configurable": {"thread_id": session_id}}


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
        "retry_counts": {"decomposer": 0, "search": 0, "summariser": 0},
        "error": None,
        "search_days_used": None,
        "selected_articles": [],
        "logs": [],
    }

    async def run_phase_1():
        print(
            f"[{session_id}] 🚀 Starting LangGraph Workflow Phase 1 (Search & Filter)..."
        )
        await graph_app.ainvoke(initial_state, config=get_config(session_id))
        print(
            f"[{session_id}] ⏳ Phase 1 paused. Waiting for human-in-the-loop selection..."
        )

    asyncio.create_task(run_phase_1())

    return {"session_id": session_id, "status": "started"}


@app.get("/api/workflow/status/{session_id}")
async def get_workflow_status(session_id: str):
    config = get_config(session_id)
    state = graph_app.get_state(config)

    if not state or not state.values:
        return {
            "session_id": session_id,
            "status": "pending",
            "current_stage": "initializing",
            "progress_percentage": 0,
            "stages": [
                {
                    "id": "understand",
                    "label": "Understanding your query",
                    "status": "running",
                },
                {
                    "id": "identify",
                    "label": "Identifying topic & domain",
                    "status": "pending",
                },
                {
                    "id": "collect",
                    "label": "Collecting source candidates",
                    "status": "pending",
                },
                {
                    "id": "filter",
                    "label": "Filtering & ranking content",
                    "status": "pending",
                },
                {
                    "id": "analyze",
                    "label": "Analyzing selected documents",
                    "status": "pending",
                },
                {
                    "id": "prepare",
                    "label": "Preparing results for review",
                    "status": "pending",
                },
            ],
            "logs": [],
        }

    next_tasks = state.next
    is_paused_for_human = "summariser" in next_tasks
    has_final_report = bool(state.values.get("final_report"))

    logs = state.values.get("logs", [])

    # Determine overall status and progress
    if has_final_report:
        status = "completed"
        progress = 100
        current_stage = "finished"
        stages = [
            {
                "id": "understand",
                "label": "Understanding your query",
                "status": "completed",
            },
            {
                "id": "identify",
                "label": "Identifying topic & domain",
                "status": "completed",
            },
            {
                "id": "collect",
                "label": "Collecting source candidates",
                "status": "completed",
            },
            {
                "id": "filter",
                "label": "Filtering & ranking content",
                "status": "completed",
            },
            {
                "id": "analyze",
                "label": "Analyzing selected documents",
                "status": "completed",
            },
            {
                "id": "prepare",
                "label": "Preparing results for review",
                "status": "completed",
            },
        ]
    elif is_paused_for_human:
        status = "completed"
        progress = 100
        current_stage = "select"
        stages = [
            {
                "id": "understand",
                "label": "Understanding your query",
                "status": "completed",
            },
            {
                "id": "identify",
                "label": "Identifying topic & domain",
                "status": "completed",
            },
            {
                "id": "collect",
                "label": "Collecting source candidates",
                "status": "completed",
            },
            {
                "id": "filter",
                "label": "Filtering & ranking content",
                "status": "completed",
            },
            {
                "id": "analyze",
                "label": "Analyzing selected documents",
                "status": "completed",
            },
            {
                "id": "prepare",
                "label": "Preparing results for review",
                "status": "completed",
            },
        ]
    else:
        # Workflow is still running — estimate progress from logs count
        n_logs = len(logs)
        progress = min(10 + n_logs * 8, 85)
        status = "running"
        current_stage = "searching"

        # Derive stage statuses based on log count heuristic
        stages = _build_running_stages(n_logs)

    return {
        "session_id": session_id,
        "status": status,
        "current_stage": current_stage,
        "progress_percentage": progress,
        "stages": stages,
        "logs": logs,
    }


def _build_running_stages(n_logs: int) -> list:
    """Build stage objects based on estimated progress from log count."""

    def s(label_idx):
        if n_logs > label_idx * 2 + 2:
            return "completed"
        elif n_logs > label_idx * 2:
            return "running"
        return "pending"

    return [
        {"id": "understand", "label": "Understanding your query", "status": s(0)},
        {"id": "identify", "label": "Identifying topic & domain", "status": s(1)},
        {"id": "collect", "label": "Collecting source candidates", "status": s(2)},
        {"id": "filter", "label": "Filtering & ranking content", "status": s(3)},
        {"id": "analyze", "label": "Analyzing selected documents", "status": s(4)},
        {"id": "prepare", "label": "Preparing results for review", "status": s(5)},
    ]


@app.get("/api/articles/{session_id}")
async def get_scored_articles(session_id: str):
    config = get_config(session_id)
    state = graph_app.get_state(config)

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
    state = graph_app.get_state(config)
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
    state = graph_app.get_state(config)

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

    return {
        "session_id": session_id,
        "report": final_report,
        "pdf_url": f"/api/pdf/download/{session_id}"
        if os.path.exists(f"report_{session_id}.pdf")
        else None,
    }


@app.post("/api/generate-pdf/{session_id}")
async def generate_pdf(session_id: str, request: GeneratePdfRequest):
    config = get_config(session_id)
    state = graph_app.get_state(config)

    if not state:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not state.next or "summariser" not in state.next:
        raise HTTPException(
            status_code=400,
            detail="Workflow is not currently paused waiting for article selection.",
        )

    print(
        f"[{session_id}] Received user selection. {len(request.selected_article_urls)} articles selected."
    )

    # Update state with the selected URLs
    graph_app.update_state(config, {"selected_articles": request.selected_article_urls})

    print(f"[{session_id}] 🚀 Starting Phase 2 (Summarise & PDF)...")
    final_output = await graph_app.ainvoke(None, config=config)

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
    state = graph_app.get_state(config)

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
