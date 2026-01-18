import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, TypedDict
from datetime import datetime
import logging

from langchain_core.messages import HumanMessage
import time

from app.agents.rca_agent import rca_app
from app.agents.fix_agent import fix_app
from app.agents.patch_agent import patch_app

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OrchestratorState(TypedDict):
    """
    Global state that orchestrates all three agents
    """
    
    trace_data: str
    codebase_root: str
    
    
    shared_memory: Dict[str, Any]
    
    
    message_history: list
    
    
    workflow_status: str
    current_agent: str


def rca_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Execute RCA Agent and update orchestrator state
    """
    logger.info("=" * 80)
    logger.info("EXECUTING RCA AGENT")
    logger.info(f"Trace data length: {len(state['trace_data'])} bytes")
    logger.info("=" * 80)
    
    # current agent
    state["current_agent"] = "RCA Agent"
    state["workflow_status"] = "running_rca"
    
    
    rca_input = {
        "messages": [HumanMessage(content=state["trace_data"])],
        "shared_memory": {
            "iteration": 0
        },
        "message_history": []
    }
    
    
    rca_result = rca_app.invoke(rca_input)
    
    
    rca_output = rca_result["shared_memory"].get("rca_result")
    
    if not rca_output:
        raise ValueError("RCA Agent failed to produce output")
    
    
    state["shared_memory"]["rca_result"] = rca_output
    state["shared_memory"]["rca_iteration"] = rca_result["shared_memory"].get("iteration", 0)
    
    
    for msg in rca_result["message_history"]:
        if "agent" not in msg:
            msg["agent"] = "RCA Agent"
        state["message_history"].append(msg)
    
    logger.info("=" * 80)
    logger.info("RCA AGENT COMPLETED")
    logger.info(f"Iterations: {rca_result['shared_memory'].get('iteration', 0)}")
    logger.info(f"Error Type: {rca_output.get('error_type')}")
    logger.info(f"Affected File: {rca_output.get('affected_file')}")
    logger.info(f"Affected Line: {rca_output.get('affected_line')}")
    logger.info(f"Root Cause: {rca_output.get('root_cause', '')}...")
    logger.info("=" * 80)
    
    return state


def fix_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Execute Fix Suggestion Agent and update orchestrator state
    """
    logger.info("=" * 80)
    logger.info("EXECUTING FIX SUGGESTION AGENT")
    logger.info("=" * 80)
    
    
    state["current_agent"] = "Fix Suggestion Agent"
    state["workflow_status"] = "running_fix"
    
    
    fix_input = {
        "messages": [HumanMessage(content="Generate fix plan based on RCA")],
        "shared_memory": {
            "rca_result": state["shared_memory"]["rca_result"],
            "iteration": 0  
        },
        "message_history": []
    }
    
    
    fix_result = fix_app.invoke(fix_input)
    
    
    fix_output = fix_result["shared_memory"].get("fix_result")
    
    if not fix_output:
        raise ValueError("Fix Agent failed to produce output")
    
    #
    state["shared_memory"]["fix_result"] = fix_output
    state["shared_memory"]["fix_iteration"] = fix_result["shared_memory"].get("iteration", 0)
    
    # Append Fix message history to global history
    for msg in fix_result["message_history"]:
        if "agent" not in msg:
            msg["agent"] = "Fix Suggestion Agent"
        state["message_history"].append(msg)
    
    logger.info("=" * 80)
    logger.info("FIX SUGGESTION AGENT COMPLETED")
    logger.info(f"Iterations: {fix_result['shared_memory'].get('iteration', 0)}")
    logger.info(f"Files to Modify: {fix_output.get('files_to_modify')}")
    logger.info(f"Patch Plan Steps: {len(fix_output.get('patch_plan', []))}")
    logger.info(f"Fix Summary: {fix_output.get('fix_summary', '')}...")
    logger.info("=" * 80)
    
    return state


def patch_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Execute Patch Generation Agent and update orchestrator state
    """
    logger.info("=" * 80)
    logger.info("EXECUTING PATCH GENERATION AGENT")
    logger.info("=" * 80)
    
    
    state["current_agent"] = "Patch Generation Agent"
    state["workflow_status"] = "running_patch"
    
    
    patch_input = {
        "messages": [HumanMessage(content="Generate patch using Fix Plan and tools")],
        "shared_memory": {
            "rca_result": state["shared_memory"]["rca_result"],
            "fix_result": state["shared_memory"]["fix_result"],
            "patch_iteration": 0
        },
        "message_history": []
    }
    
    
    patch_result = patch_app.invoke(patch_input)
    
    
    patch_output = patch_result["shared_memory"].get("patch_result")
    
    if not patch_output:
        raise ValueError("Patch Agent failed to produce output")
    
    
    state["shared_memory"]["patch_result"] = patch_output
    state["shared_memory"]["patch_iteration"] = patch_result["shared_memory"].get("patch_iteration", 0)
    
    
    for msg in patch_result["message_history"]:
        if "agent" not in msg:
            msg["agent"] = "Patch Generation Agent"
        state["message_history"].append(msg)
    
    logger.info("=" * 80)
    logger.info("PATCH GENERATION AGENT COMPLETED")
    logger.info(f"Iterations: {patch_result['shared_memory'].get('patch_iteration', 0)}")
    logger.info(f"Patch File Created: {patch_output.get('success', False)}")
    logger.info(f"Patch File Path: {patch_output.get('patch_file', 'N/A')}")
    logger.info(f"Original File: {patch_output.get('original_file', 'N/A')}")
    logger.info("=" * 80)
    
    state["workflow_status"] = "completed"
    
    return state


def build_orchestration_graph():
    """
    Build the orchestration graph that chains all three agents
    """
    from langgraph.graph import StateGraph, START, END
    
    graph = StateGraph(OrchestratorState)
    
    
    graph.add_node("rca_agent", rca_agent_node)
    graph.add_node("fix_agent", fix_agent_node)
    graph.add_node("patch_agent", patch_agent_node)
    
    
    graph.add_edge(START, "rca_agent")
    graph.add_edge("rca_agent", "fix_agent")
    graph.add_edge("fix_agent", "patch_agent")
    graph.add_edge("patch_agent", END)
    
    return graph.compile()


def main():
    """
    Main orchestration function that runs the complete multi-agent workflow
    """

    CODEBASE_ROOT = os.getenv("CODEBASE_ROOT")
    TRACE_FILE = os.getenv("TRACE_FILE")
    OUTPUT_DIR = Path("output")
    

    os.environ["CODEBASE_ROOT"] = CODEBASE_ROOT
    
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("MULTI-AGENT RCA SYSTEM STARTED")
    logger.info(f"Codebase Root: {CODEBASE_ROOT}")
    logger.info(f"Trace File: {TRACE_FILE}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info("=" * 80)
    

    try:
        with open(TRACE_FILE, "r", encoding="utf-8") as f:
            trace_data = f.read()  
        logger.info(f"Trace file loaded successfully - Size: {len(trace_data)} bytes")
        logger.info(f"Preview: {trace_data}...")
    except Exception as e:
        logger.error(f"Failed to load trace file: {str(e)}")
        return
    
    
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
            "timestamp": datetime.now().isoformat(),
            "trace_file": TRACE_FILE,
            "codebase_root": CODEBASE_ROOT
        }],
        "workflow_status": "initialized",
        "current_agent": "None"
    }
    
    
    try:
        
        orchestrator = build_orchestration_graph()
        
        logger.info("=" * 80)
        logger.info("STARTING AGENT ORCHESTRATION")
        logger.info("=" * 80)
        
        
        final_state = orchestrator.invoke(initial_state)
        
        
        final_state["message_history"].append({
            "event": "workflow_completed",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })
        
        logger.info("=" * 80)
        logger.info("AGENT ORCHESTRATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        import traceback
        logger.error("=" * 80)
        logger.error("ORCHESTRATION FAILED")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Error Type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        logger.error("=" * 80)
        
        
        initial_state["message_history"].append({
            "event": "workflow_failed",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        })
        final_state = initial_state
        final_state["workflow_status"] = "failed"
    
    
    message_history_path = OUTPUT_DIR / "message_history.json"
    with open(message_history_path, "w", encoding="utf-8") as f:
        json.dump(final_state["message_history"], f, indent=2, ensure_ascii=False)
    logger.info(f"Message history saved to: {message_history_path}")
    
    
    shared_memory_path = OUTPUT_DIR / "shared_memory.json"
    with open(shared_memory_path, "w", encoding="utf-8") as f:
        json.dump(final_state["shared_memory"], f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Shared memory saved to: {shared_memory_path}")
    
    #
    logger.info("=" * 80)
    logger.info("WORKFLOW SUMMARY")
    logger.info(f"Status: {final_state['workflow_status']}")
    logger.info(f"Total Messages: {len(final_state['message_history'])}")
    logger.info(f"RCA Completed: {final_state['shared_memory'].get('rca_result') is not None}")
    logger.info(f"Fix Completed: {final_state['shared_memory'].get('fix_result') is not None}")
    logger.info(f"Patch Completed: {final_state['shared_memory'].get('patch_result') is not None}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info("=" * 80)
    
    
    if final_state["shared_memory"].get("patch_result"):
        patch_info = final_state["shared_memory"]["patch_result"]
        logger.info("=" * 80)
        logger.info("PATCH FILE GENERATED")
        logger.info(f"Success: {patch_info.get('success')}")
        logger.info(f"Patch File: {patch_info.get('patch_file')}")
        logger.info(f"Original File: {patch_info.get('original_file')}")
        logger.info("=" * 80)
    
    
    logger.info("")
    logger.info("=" * 80)
    logger.info(" Multi-Agent RCA System execution completed!")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f" Message History: {message_history_path}")
    logger.info(f"Shared Memory: {shared_memory_path}")
    logger.info(f"Log File: orchestrator.log")
    
    if final_state["shared_memory"].get("patch_result"):
        patch_info = final_state["shared_memory"]["patch_result"]
        if patch_info.get("success"):
            logger.info(f"Patch File: {patch_info.get('patch_file')}")
        else:
            logger.info(f"Patch Generation Failed: {patch_info.get('error', 'Unknown error')}")
    
    logger.info("=" * 80)
    logger.info("")


if __name__ == "__main__":
    main()