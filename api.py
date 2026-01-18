"""
FastAPI Backend for Multi-Agent RCA System
Provides REST API interface for Root Cause Analysis, Fix Suggestion, and Patch Generation
"""
import os
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.main import OrchestratorState


# ============================================================================
# FastAPI App Initialization
# ============================================================================
app = FastAPI(
    title="Multi-Agent RCA System API",
    description="AI-powered Root Cause Analysis, Fix Suggestion, and Patch Generation System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow
workflow = OrchestratorState()

# Track active analyses
active_analyses: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class AnalyzeRequest(BaseModel):
    """Request model for analysis"""
    trace_file: str = Field(..., description="Path to the error trace JSON file")
    codebase_root: Optional[str] = Field(None, description="Root directory of the codebase")
    
    class Config:
        json_schema_extra = {
            "example": {
                "trace_file": "trace_1.json",
                "codebase_root": "/path/to/codebase"
            }
        }


class AnalyzeResponse(BaseModel):
    """Response model for analysis"""
    success: bool
    message: str
    analysis_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class RCAResult(BaseModel):
    """RCA analysis result"""
    error_type: Optional[str]
    error_message: Optional[str]
    root_cause: Optional[str]
    affected_file: Optional[str]
    affected_line: Optional[int]


class FixResult(BaseModel):
    """Fix suggestion result"""
    fix_summary: Optional[str]
    files_to_modify: Optional[List[str]]
    patch_plan: Optional[List[str]]
    safety_considerations: Optional[str]


class PatchResult(BaseModel):
    """Patch generation result"""
    patch_file_path: Optional[str]
    patch_success: bool
    original_file: Optional[str]
    replacements_made: Optional[int]


class FullAnalysisResult(BaseModel):
    """Complete analysis result"""
    workflow_complete: bool
    workflow_success: bool
    total_iterations: int
    rca: RCAResult
    fix: FixResult
    patch: PatchResult
    agents_executed: List[str]
    message_count: int


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint - API information
    """
    return {
        "service": "Multi-Agent RCA System",
        "version": "1.0.0",
        "description": "AI-powered debugging and patch generation system",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /status": "System status and capabilities",
            "POST /analyze": "Run complete RCA analysis workflow",
            "POST /analyze/async": "Run analysis asynchronously",
            "GET /results/latest": "Get latest analysis results",
            "GET /results/{analysis_id}": "Get specific analysis results",
            "GET /download/patch/{filename}": "Download generated patch file",
            "GET /agents": "List available agents",
            "GET /tools": "List available tools"
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Multi-Agent RCA System",
        "version": "1.0.0"
    }


@app.get("/status")
async def get_status():
    """
    Get detailed system status and capabilities
    """
    return {
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "rca_agent": {
                "name": "Root Cause Analysis Agent",
                "description": "Analyzes error traces and identifies root causes",
                "capabilities": [
                    "Stack trace analysis",
                    "Code inspection",
                    "Dependency checking",
                    "File system exploration"
                ]
            },
            "fix_agent": {
                "name": "Fix Suggestion Agent",
                "description": "Generates actionable fix plans",
                "capabilities": [
                    "Fix plan generation",
                    "Safety analysis",
                    "File modification planning"
                ]
            },
            "patch_agent": {
                "name": "Patch Generation Agent",
                "description": "Creates actual code patches",
                "capabilities": [
                    "Code patch generation",
                    "File creation",
                    "Minimal code changes"
                ]
            }
        },
        "tools": [
            "read_file - Read source code files",
            "get_project_directory - Explore project structure",
            "check_dependency - Verify dependencies",
            "create_patch_file - Generate patch files"
        ],
        "active_analyses": len(active_analyses),
        "output_directory": str(workflow.output_dir)
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_trace(request: AnalyzeRequest):
    """
    Run complete multi-agent RCA analysis (synchronous)
    
    This endpoint runs the full workflow:
    1. RCA Agent analyzes the error
    2. Fix Agent generates fix plan
    3. Patch Agent creates the patch
    
    Args:
        request: AnalyzeRequest with trace_file and optional codebase_root
    
    Returns:
        AnalyzeResponse with complete results
    """
    # Generate analysis ID
    analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Validate trace file exists
        if not os.path.exists(request.trace_file):
            raise HTTPException(
                status_code=404,
                detail=f"Trace file not found: {request.trace_file}"
            )
        
        print(f"\n{'='*80}")
        print(f"🚀 Starting Analysis: {analysis_id}")
        print(f"📄 Trace File: {request.trace_file}")
        print(f"📁 Codebase: {request.codebase_root or os.getenv('CODEBASE_ROOT', '.')}")
        print(f"{'='*80}\n")
        
        # Track analysis
        active_analyses[analysis_id] = {
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "trace_file": request.trace_file
        }
        
        # Run workflow
        final_state = workflow.run(
            trace_file=request.trace_file,
            codebase_root=request.codebase_root
        )
        
        # Update analysis status
        active_analyses[analysis_id]["status"] = "completed"
        active_analyses[analysis_id]["end_time"] = datetime.now().isoformat()
        active_analyses[analysis_id]["results"] = final_state
        
        # Prepare response
        response_data = {
            "analysis_id": analysis_id,
            "workflow": {
                "complete": final_state.get("workflow_complete"),
                "success": final_state.get("workflow_success"),
                "status": final_state.get("workflow_status"),
                "total_iterations": final_state.get("total_iterations"),
                "agents_executed": final_state.get("agents_executed", [])
            },
            "rca": {
                "error_type": final_state.get("error_type"),
                "error_message": final_state.get("error_message"),
                "root_cause": final_state.get("root_cause"),
                "affected_file": final_state.get("affected_file"),
                "affected_line": final_state.get("affected_line")
            },
            "fix": {
                "summary": final_state.get("fix_summary"),
                "files_to_modify": final_state.get("files_to_modify", []),
                "patch_plan": final_state.get("patch_plan", []),
                "safety_considerations": final_state.get("safety_considerations")
            },
            "patch": {
                "file_path": final_state.get("patch_file_path"),
                "success": final_state.get("patch_success", False),
                "original_file": final_state.get("original_file"),
                "replacements_made": final_state.get("replacements_made", 0)
            },
            "metadata": {
                "message_count": final_state.get("message_count"),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        print(f"\n{'='*80}")
        print(f"✅ Analysis Complete: {analysis_id}")
        print(f"📊 Status: {final_state.get('workflow_status')}")
        if final_state.get("patch_file_path"):
            print(f"🔧 Patch File: {final_state.get('patch_file_path')}")
        print(f"{'='*80}\n")
        
        return AnalyzeResponse(
            success=True,
            message="Analysis completed successfully",
            analysis_id=analysis_id,
            data=response_data
        )
        
    except Exception as e:
        # Update analysis status
        active_analyses[analysis_id]["status"] = "failed"
        active_analyses[analysis_id]["error"] = str(e)
        active_analyses[analysis_id]["end_time"] = datetime.now().isoformat()
        
        print(f"\n❌ Analysis Failed: {analysis_id}")
        print(f"Error: {str(e)}\n")
        
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze/async")
async def analyze_trace_async(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """
    Run analysis asynchronously in the background
    
    Returns immediately with an analysis ID.
    Use GET /results/{analysis_id} to check status and get results.
    """
    analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Validate trace file
    if not os.path.exists(request.trace_file):
        raise HTTPException(
            status_code=404,
            detail=f"Trace file not found: {request.trace_file}"
        )
    
    # Initialize analysis tracking
    active_analyses[analysis_id] = {
        "status": "queued",
        "start_time": datetime.now().isoformat(),
        "trace_file": request.trace_file
    }
    
    # Add to background tasks
    background_tasks.add_task(
        run_analysis_background,
        analysis_id,
        request.trace_file,
        request.codebase_root
    )
    
    return {
        "success": True,
        "message": "Analysis queued successfully",
        "analysis_id": analysis_id,
        "status": "queued",
        "check_status_url": f"/results/{analysis_id}"
    }


async def run_analysis_background(
    analysis_id: str,
    trace_file: str,
    codebase_root: Optional[str]
):
    """Background task to run analysis"""
    try:
        active_analyses[analysis_id]["status"] = "running"
        
        final_state = workflow.run(
            trace_file=trace_file,
            codebase_root=codebase_root
        )
        
        active_analyses[analysis_id]["status"] = "completed"
        active_analyses[analysis_id]["end_time"] = datetime.now().isoformat()
        active_analyses[analysis_id]["results"] = final_state
        
    except Exception as e:
        active_analyses[analysis_id]["status"] = "failed"
        active_analyses[analysis_id]["error"] = str(e)
        active_analyses[analysis_id]["end_time"] = datetime.now().isoformat()


@app.get("/results/latest")
async def get_latest_results():
    """
    Get the most recent analysis results from saved files
    """
    try:
        results = workflow.get_latest_results()
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No results found. Run an analysis first."
            )
        
        return {
            "success": True,
            "data": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """
    Get results for a specific analysis ID
    """
    if analysis_id not in active_analyses:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis ID not found: {analysis_id}"
        )
    
    analysis = active_analyses[analysis_id]
    
    response = {
        "analysis_id": analysis_id,
        "status": analysis["status"],
        "start_time": analysis["start_time"],
        "trace_file": analysis["trace_file"]
    }
    
    if "end_time" in analysis:
        response["end_time"] = analysis["end_time"]
    
    if analysis["status"] == "completed":
        response["results"] = analysis.get("results")
    elif analysis["status"] == "failed":
        response["error"] = analysis.get("error")
    
    return response


@app.get("/download/patch/{filename}")
async def download_patch(filename: str):
    """
    Download a generated patch file
    """
    file_path = workflow.output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Patch file not found: {filename}"
        )
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="text/x-python"
    )


@app.get("/agents")
async def list_agents():
    """
    List all available agents and their capabilities
    """
    return {
        "agents": [
            {
                "id": "rca_agent",
                "name": "Root Cause Analysis Agent",
                "description": "Analyzes error traces and identifies root causes",
                "tools": [
                    "read_file",
                    "get_project_directory",
                    "check_dependency"
                ],
                "outputs": [
                    "error_type",
                    "error_message",
                    "root_cause",
                    "affected_file",
                    "affected_line"
                ]
            },
            {
                "id": "fix_agent",
                "name": "Fix Suggestion Agent",
                "description": "Generates actionable fix plans based on RCA",
                "tools": [
                    "read_file",
                    "get_project_directory"
                ],
                "outputs": [
                    "fix_summary",
                    "files_to_modify",
                    "patch_plan",
                    "safety_considerations"
                ]
            },
            {
                "id": "patch_agent",
                "name": "Patch Generation Agent",
                "description": "Creates actual code patches",
                "tools": [
                    "read_file",
                    "create_patch_file"
                ],
                "outputs": [
                    "patch_file_path",
                    "patch_success",
                    "replacements_made"
                ]
            }
        ]
    }


@app.get("/tools")
async def list_tools():
    """
    List all available tools
    """
    return {
        "tools": [
            {
                "name": "read_file",
                "description": "Read source code files from the codebase",
                "parameters": ["file_path"]
            },
            {
                "name": "get_project_directory",
                "description": "Explore and map project directory structure",
                "parameters": ["directory_path"]
            },
            {
                "name": "check_dependency",
                "description": "Verify installed packages and dependencies",
                "parameters": ["package_name"]
            },
            {
                "name": "create_patch_file",
                "description": "Generate patched code files",
                "parameters": ["original_file_path", "fixed_content"]
            }
        ]
    }


@app.delete("/results/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Delete a specific analysis from active analyses
    """
    if analysis_id not in active_analyses:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis ID not found: {analysis_id}"
        )
    
    del active_analyses[analysis_id]
    
    return {
        "success": True,
        "message": f"Analysis {analysis_id} deleted successfully"
    }


@app.get("/analyses")
async def list_analyses():
    """
    List all tracked analyses
    """
    return {
        "total": len(active_analyses),
        "analyses": {
            aid: {
                "status": data["status"],
                "start_time": data["start_time"],
                "trace_file": data["trace_file"]
            }
            for aid, data in active_analyses.items()
        }
    }


# ============================================================================
# Run Server
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 Multi-Agent RCA System API Server")
    print("="*80)
    print(f"📍 Server: http://0.0.0.0:8000")
    print(f"📚 API Docs: http://0.0.0.0:8000/docs")
    print(f"📖 ReDoc: http://0.0.0.0:8000/redoc")
  
    print("="*80 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )