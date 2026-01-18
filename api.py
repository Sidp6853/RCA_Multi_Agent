import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv

# Load env
load_dotenv()

# Import orchestrator builder + state
from app.main import build_orchestration_graph, OrchestratorState


# -------------------------
# FastAPI App Init
# -------------------------



app = FastAPI(
    title="Multi-Agent RCA System API",
    description="Root Cause Analysis + Fix + Patch Generation API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Load Orchestrator ONCE
# -------------------------

print("🚀 Initializing Orchestrator Graph...")
orchestrator = build_orchestration_graph()
print("✅ Orchestrator Ready")


# -------------------------
# Config
# -------------------------

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CODEBASE_ROOT = os.getenv(
    "CODEBASE_ROOT",
    r"D:\Siddhi\projects\RCA-Agent\codebase"
)

os.environ["CODEBASE_ROOT"] = CODEBASE_ROOT


# -------------------------
# Request / Response Models
# -------------------------

class AnalyzeRequest(BaseModel):
    trace_file_path: Optional[str] = None   # Use existing trace file
    trace_text: Optional[str] = None        # Or raw trace content


class AnalyzeResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# -------------------------
# Health Check
# -------------------------

@app.get("/")
def root():
    return {
        "service": "Multi-Agent RCA System",
        "status": "running",
        "agents": ["RCA", "Fix", "Patch"]
    }


@app.get("/status")
def status():
    return {
        "status": "operational",
        "codebase_root": CODEBASE_ROOT
    }


# -------------------------
# MAIN WORKFLOW ENDPOINT
# -------------------------

@app.get("/")
def home():
    return {"status": "RCA Agent API Running"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Run complete RCA -> Fix -> Patch workflow
    """

    try:
        # -------------------------
        # Load Trace Data
        # -------------------------

        if request.trace_text:
            trace_data = request.trace_text

        elif request.trace_file_path:
            if not os.path.exists(request.trace_file_path):
                raise HTTPException(
                    status_code=404,
                    detail="Trace file not found"
                )

            with open(request.trace_file_path, "r", encoding="utf-8") as f:
                trace_data = f.read()

        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either trace_text or trace_file_path"
            )

        # -------------------------
        # Build Initial State
        # -------------------------

        initial_state: OrchestratorState = {
            "trace_data": trace_data,
            "codebase_root": CODEBASE_ROOT,
            "shared_memory": {
                "workflow_start_time": datetime.now().isoformat(),
                "rca_result": None,
                "fix_result": None,
                "patch_result": None
            },
            "message_history": [{
                "event": "workflow_started",
                "timestamp": datetime.now().isoformat()
            }],
            "workflow_status": "initialized",
            "current_agent": "None"
        }

        # -------------------------
        # Execute Workflow
        # -------------------------

        final_state = orchestrator.invoke(initial_state)

        # -------------------------
        # Save Outputs
        # -------------------------

        history_file = OUTPUT_DIR / "message_history.json"
        memory_file = OUTPUT_DIR / "shared_memory.json"

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(final_state["message_history"], f, indent=2)

        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(final_state["shared_memory"], f, indent=2, default=str)

        # -------------------------
        # Prepare API Response
        # -------------------------

        response_payload = {
            "workflow_status": final_state["workflow_status"],

            "rca": final_state["shared_memory"].get("rca_result"),
            "fix": final_state["shared_memory"].get("fix_result"),
            "patch": final_state["shared_memory"].get("patch_result"),

            "message_count": len(final_state["message_history"])
        }

        return AnalyzeResponse(
            success=True,
            message="Workflow completed successfully",
            data=response_payload
        )

    except Exception as e:
        return AnalyzeResponse(
            success=False,
            message="Workflow execution failed",
            error=str(e)
        )


# -------------------------
# RESULTS ENDPOINT
# -------------------------

@app.get("/results")
def get_latest_results():
    """
    Fetch last saved workflow output
    """

    try:
        history_file = OUTPUT_DIR / "message_history.json"
        memory_file = OUTPUT_DIR / "shared_memory.json"

        results = {}

        if history_file.exists():
            with open(history_file, "r") as f:
                results["message_history"] = json.load(f)

        if memory_file.exists():
            with open(memory_file, "r") as f:
                results["shared_memory"] = json.load(f)

        return {
            "success": True,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# Server Runner
# -------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
