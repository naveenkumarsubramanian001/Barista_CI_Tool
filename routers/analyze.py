import os
import secrets
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel
import fitz  # PyMuPDF

router = APIRouter(prefix="/api/analyze", tags=["analyze"])


async def _get_graph_state(graph, config):
    """Support both async LangGraph APIs and sync test doubles."""
    if hasattr(graph, "aget_state"):
        return await graph.aget_state(config)
    return graph.get_state(config)

# We'll import analyzer_app once we build it in api.py or graph/analyzer_workflow.py
# from api import analyzer_app, get_config

@router.post("/upload")
async def upload_document(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    session_id = f"analyzer_{secrets.token_hex(8)}"
    
    content = await file.read()
    
    if file.filename.lower().endswith(".pdf"):
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            extracted_text = text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")
    else:
        try:
            extracted_text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please use PDF or UTF-8 text.")
            
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No readable text found in document.")

    import asyncio

    analyzer_app = request.app.state.analyzer_app

    def get_config(session_id: str):
        return {"configurable": {"thread_id": session_id}}
    
    initial_state = {
        "session_id": session_id,
        "uploaded_text": extracted_text,
        "product_profile": None,
        "discovered_competitors": [],
        "competitor_data": {},
        "final_report": None,
        "logs": ["📥 Document parsed successfully. Initializing analysis..."],
        "workflow_status": "extracting",
        "progress_percentage": 10,
        "error": None
    }
    
    async def run_analysis():
        try:
            print(f"[{session_id}] 🚀 Starting Analyzer Workflow...")
            final_output = await analyzer_app.ainvoke(initial_state, config=get_config(session_id))
            print(f"[{session_id}] ✅ Analyzer Workflow Completed.")
            
            if final_output and final_output.get("final_report"):
                import json
                report_json_name = f"analyze_report_{session_id}.json"
                report_pdf_name = f"analyze_report_{session_id}.pdf"
                
                with open(report_json_name, "w") as f:
                    json.dump(final_output["final_report"], f, indent=2)
                
                from utils.comparative_pdf_report import generate_comparative_pdf
                generate_comparative_pdf(report_json_name, report_pdf_name)
                print(f"[{session_id}] 🖨️ PDF generated successfully.")
            else:
                print(f"[{session_id}] ⚠️ Workflow completed but no final_report found.")
                error = final_output.get("error") if final_output else None
                if error:
                    print(f"[{session_id}] ❌ Error recorded in state: {error}")
                
        except Exception as e:
            print(f"[{session_id}] ❌ Analyzer Workflow Failed: {e}")
            
    asyncio.create_task(run_analysis())
    
    return {"session_id": session_id, "status": "started", "message": "Document uploaded successfully"}

@router.get("/status/{session_id}")
async def get_analyze_status(request: Request, session_id: str):
    analyzer_app = request.app.state.analyzer_app

    def get_config(current_session_id: str):
        return {"configurable": {"thread_id": current_session_id}}

    config = get_config(session_id)
    state = await _get_graph_state(analyzer_app, config)
    
    if not state or not state.values:
        return {
            "session_id": session_id,
            "status": "initializing",
            "progress_percentage": 0,
            "logs": []
        }
        
    has_report = state.values.get("final_report") is not None
    import os
    pdf_exists = os.path.exists(f"analyze_report_{session_id}.pdf")
    
    if has_report and pdf_exists:
        status_str = "completed"
        progress = 100
    elif state.values.get("error"):
        status_str = "failed"
        progress = state.values.get("progress_percentage", 10)
    else:
        status_str = state.values.get("workflow_status", "running")
        progress = state.values.get("progress_percentage", 10)
        
    return {
        "session_id": session_id,
        "status": status_str,
        "progress_percentage": progress,
        "logs": state.values.get("logs", []),
        "report_data": state.values.get("final_report") if has_report else None
    }
    
from fastapi.responses import FileResponse

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
