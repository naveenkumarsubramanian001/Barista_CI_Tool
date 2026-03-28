import asyncio
import json
import os
import secrets

import fitz  # PyMuPDF
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from models.schemas import AnalyzeStatusResponse, ApiError

router = APIRouter(prefix="/api/analyze", tags=["analyze"])


def _resolve_runtime(request: Request):
    analyzer_app = getattr(request.app.state, "analyzer_app", None)
    get_config = getattr(request.app.state, "get_config", None)
    if analyzer_app is None or get_config is None:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "ANALYZER_NOT_CONFIGURED",
                "message": "Analyzer runtime is not configured on app state.",
            },
        )
    return analyzer_app, get_config


@router.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    session_id = f"analyzer_{secrets.token_hex(8)}"

    content = await file.read()

    if file.filename.lower().endswith(".pdf"):
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            extracted_text = text
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}")
    else:
        try:
            extracted_text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please use PDF or UTF-8 text.",
            )

    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in document.")

    analyzer_app, get_config = _resolve_runtime(request)

    initial_state = {
        "session_id": session_id,
        "uploaded_text": extracted_text,
        "product_profile": None,
        "discovered_competitors": [],
        "competitor_data": {},
        "final_report": None,
        "logs": ["Document parsed successfully. Initializing analysis..."],
        "workflow_status": "extracting",
        "progress_percentage": 10,
        "error": None,
    }

    async def run_analysis():
        try:
            final_output = await analyzer_app.ainvoke(initial_state, config=get_config(session_id))

            if final_output and final_output.get("final_report"):
                report_json_name = f"analyze_report_{session_id}.json"
                report_pdf_name = f"analyze_report_{session_id}.pdf"

                with open(report_json_name, "w", encoding="utf-8") as out:
                    json.dump(final_output["final_report"], out, indent=2)

                from utils.comparative_pdf_report import generate_comparative_pdf

                generate_comparative_pdf(report_json_name, report_pdf_name)
        except Exception as exc:
            # Persist a failure breadcrumb in state if available.
            try:
                analyzer_app.update_state(
                    get_config(session_id),
                    {
                        "error": f"Analyzer workflow failed: {exc}",
                        "workflow_status": "failed",
                    },
                )
            except Exception:
                pass

    asyncio.create_task(run_analysis())

    return {
        "api_version": "v1",
        "session_id": session_id,
        "status": "started",
        "message": "Document uploaded successfully",
        "error": None,
    }


@router.get("/status/{session_id}", response_model=AnalyzeStatusResponse)
async def get_analyze_status(request: Request, session_id: str):
    analyzer_app, get_config = _resolve_runtime(request)
    config = get_config(session_id)
    state = analyzer_app.get_state(config)

    if not state or not state.values:
        return AnalyzeStatusResponse(
            session_id=session_id,
            status="initializing",
            progress_percentage=0,
            logs=[],
            error=None,
        ).model_dump()

    has_report = state.values.get("final_report") is not None
    pdf_exists = os.path.exists(f"analyze_report_{session_id}.pdf")

    if has_report and pdf_exists:
        status_str = "completed"
        progress = 100
        error_payload = None
    elif state.values.get("error"):
        status_str = "failed"
        progress = state.values.get("progress_percentage", 10)
        error_payload = ApiError(code="ANALYZER_WORKFLOW_FAILED", message=str(state.values.get("error")))
    else:
        status_str = state.values.get("workflow_status", "running")
        progress = state.values.get("progress_percentage", 10)
        error_payload = None

    return AnalyzeStatusResponse(
        session_id=session_id,
        status=status_str,
        progress_percentage=progress,
        logs=state.values.get("logs", []),
        report_data=state.values.get("final_report") if has_report else None,
        error=error_payload,
    ).model_dump()


@router.get("/download/{session_id}")
async def download_analyze_pdf(session_id: str):
    file_path = f"analyze_report_{session_id}.pdf"
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            filename="Comparative_Intelligence_Report.pdf",
            media_type="application/pdf",
        )
    raise HTTPException(status_code=404, detail="PDF not found on disk.")
