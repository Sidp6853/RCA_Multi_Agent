import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage


from app.workflow import pipeline, PipelineState

load_dotenv()

#
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initializing FastAPI app
app = FastAPI(
    title="Multi-Agent RCA System API",
    description="API for Root Cause Analysis, Fix Suggestion, and Patch Generation",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class AnalyzeRequest(BaseModel):
    """Request model for RCA analysis"""
    trace_file_path: str = Field(
        ...,
        description="Path to the error trace JSON file",
    )
    codebase_root: str = Field(
        ...,
        description="Root directory of the codebase to analyze",
    )

class AnalyzeResponse(BaseModel):
    """Response model for analysis"""
    success: bool
    message: str
    results: Optional[Dict[str, Any]] = None
    output_files: Optional[Dict[str, str]] = None
    error: Optional[str] = None



@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_codebase(request: AnalyzeRequest):
    """
    Run complete multi-agent RCA workflow: Root Cause Analysis → Fix Suggestion → Patch Generation
    
    Args:
        request: AnalyzeRequest containing trace_file_path and codebase_root
    
    Returns:
        AnalyzeResponse with complete workflow results
    """
    try:
        # Validate inputs
        if not os.path.exists(request.trace_file_path):
            raise HTTPException(
                status_code=404,
                detail=f"Trace file not found: {request.trace_file_path}"
            )
        
        if not os.path.exists(request.codebase_root):
            raise HTTPException(
                status_code=404,
                detail=f"Codebase root not found: {request.codebase_root}"
            )
        
        
        os.environ["CODEBASE_ROOT"] = request.codebase_root
        
        logger.info("=" * 80)
        logger.info("API Request received")
        logger.info(f"Trace File: {request.trace_file_path}")
        logger.info(f"Codebase Root: {request.codebase_root}")
        logger.info("=" * 80)
        
        # Load trace file
        try:
            with open(request.trace_file_path, "r", encoding="utf-8") as f:
                trace_data = f.read()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load trace file: {str(e)}"
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("output") / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # LangGraph thread ID
        thread_id = f"{timestamp}_{hash(request.trace_file_path) % 10000}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run the workflow pipeline
        logger.info("Starting workflow pipeline...")

        initial_state = PipelineState()
        initial_state.messages = [HumanMessage(content=trace_data)]
        
        final_state = pipeline.invoke(initial_state, config=config)
        final_state = dict(final_state)



        history_state = pipeline.get_state(config)
        full_history = history_state.values["messages"]

        complete_trace = {
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
            "input": {
                "trace_file": request.trace_file_path,
                "codebase_root": request.codebase_root
            },
            "complete_message_history": [
                {
                    "role": msg.role if hasattr(msg, 'role') else "unknown",
                    "content": msg.content[:500] + "..." if len(msg.content) > 500 else msg.content,
                    "tool_calls": getattr(msg, "tool_calls", []),
                    "name": getattr(msg, "name", None)
                } for msg in full_history
            ],
            
                
                "rca_result": final_state.get("rca_result"),
                "fix_result": final_state.get("fix_result"),
                "patch_result": final_state.get("patch_result"),
                
            

            "stats": {
                "total_messages": len(full_history),
                "tool_calls": len([m for m in full_history if getattr(m, "tool_calls", [])]),
                "success": bool(final_state.patch_result and final_state.patch_result.get("success"))
            }
        }

        trace_path = output_dir / "message_history.json"
        with open(trace_path, "w") as f:
            json.dump(complete_trace, f, indent=2, default=str)
        
        
        
       
        if final_state.get("error"):
            logger.error(f"Workflow failed: {final_state['error']}")
            raise HTTPException(
                status_code=500,
                detail=f"Workflow execution failed: {final_state['error']}"
            )
        
        
        response_data = {
            "rca": {
                "error_type": final_state.get("rca_result", {}).get("error_type") if final_state.get("rca_result") else None,
                "error_message": final_state.get("rca_result", {}).get("error_message") if final_state.get("rca_result") else None,
                "root_cause": final_state.get("rca_result", {}).get("root_cause") if final_state.get("rca_result") else None,
                "affected_file": final_state.get("rca_result", {}).get("affected_file") if final_state.get("rca_result") else None,
                "affected_line": final_state.get("rca_result", {}).get("affected_line") if final_state.get("rca_result") else None,
            },
            "fix": {
                "fix_summary": final_state.get("fix_result", {}).get("fix_summary") if final_state.get("fix_result") else None,
                "files_to_modify": final_state.get("fix_result", {}).get("files_to_modify") if final_state.get("fix_result") else None,
                "patch_plan": final_state.get("fix_result", {}).get("patch_plan") if final_state.get("fix_result") else None,
                "safety_considerations": final_state.get("fix_result", {}).get("safety_considerations") if final_state.get("fix_result") else None,
            },
            "patch": {
                "success": final_state.get("patch_result", {}).get("success") if final_state.get("patch_result") else False,
                "patch_file": final_state.get("patch_result", {}).get("patch_file") if final_state.get("patch_result") else None,
                "original_file": final_state.get("patch_result", {}).get("original_file") if final_state.get("patch_result") else None,
            }
        }
        
        # Save outputs
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        
        # Save RCA result
        if final_state.get("rca_result"):
            rca_path = output_dir / "shared_memory.json"
            with open(rca_path, "w") as f:
                json.dump(final_state["rca_result"], f, indent=2)
            output_files["rca_result"] = str(rca_path)
        
        # Save Fix result
        if final_state.get("fix_result"):
            fix_path = output_dir / "shared_memory.json"
            with open(fix_path, "w") as f:
                json.dump(final_state["fix_result"], f, indent=2)
            output_files["fix_result"] = str(fix_path)
        
        # Save Patch result
        if final_state.get("patch_result"):
            patch_path = output_dir / "shared_memory.json"
            with open(patch_path, "w") as f:
                json.dump(final_state["patch_result"], f, indent=2, default=str)
            output_files["patch_result"] = str(patch_path)
        
        logger.info("=" * 80)
        logger.info("Workflow completed successfully")
        logger.info(f"Patch file: {final_state.get('patch_result', {}).get('patch_file', 'N/A')}")
        logger.info("=" * 80)
        
        return AnalyzeResponse(
            success=True,
            message="Workflow completed successfully",
            results=response_data,
            output_files=output_files
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_msg = str(e)
        logger.error(f"API Error: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Return 500 error status code
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {error_msg}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Multi-Agent RCA System"}



if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=False, 
        log_level="info"
    )