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

# Configure logging
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
    # Input data
    trace_data: str
    codebase_root: str
    
    # Shared memory across all agents
    shared_memory: Dict[str, Any]
    
    # Complete message history from all agents
    message_history: list
    
    # Workflow tracking
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
    
    # Mark current agent
    state["current_agent"] = "RCA Agent"
    state["workflow_status"] = "running_rca"
    
    # Prepare RCA agent input state (matches RCAState structure)
    rca_input = {
        "messages": [HumanMessage(content=state["trace_data"])],
        "shared_memory": {
            "iteration": 0
        },
        "message_history": []
    }
    
    # Execute RCA agent
    rca_result = rca_app.invoke(rca_input)
    
    # Extract RCA output from agent's shared memory
    rca_output = rca_result["shared_memory"].get("rca_result")
    
    if not rca_output:
        raise ValueError("RCA Agent failed to produce output")
    
    # Update orchestrator's shared memory with RCA results
    state["shared_memory"]["rca_result"] = rca_output
    state["shared_memory"]["rca_iteration"] = rca_result["shared_memory"].get("iteration", 0)
    
    # Append RCA message history to global history
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
    logger.info(f"Root Cause: {rca_output.get('root_cause', '')[:200]}...")
    logger.info("=" * 80)
    
    return state


def fix_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Execute Fix Suggestion Agent and update orchestrator state
    """
    logger.info("=" * 80)
    logger.info("EXECUTING FIX SUGGESTION AGENT")
    logger.info("=" * 80)
    
    # Mark current agent
    state["current_agent"] = "Fix Suggestion Agent"
    state["workflow_status"] = "running_fix"
    
    # Prepare Fix agent input state (matches FixState structure)
    fix_input = {
        "messages": [HumanMessage(content="Generate fix plan based on RCA")],
        "shared_memory": {
            "rca_result": state["shared_memory"]["rca_result"],
            "iteration": 0  # Fix agent uses "iteration" not "fix_iteration"
        },
        "message_history": []
    }
    
    # Execute Fix agent
    fix_result = fix_app.invoke(fix_input)
    
    # Extract Fix output from agent's shared memory
    fix_output = fix_result["shared_memory"].get("fix_result")
    
    if not fix_output:
        raise ValueError("Fix Agent failed to produce output")
    
    # Update orchestrator state
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
    logger.info(f"Fix Summary: {fix_output.get('fix_summary', '')[:200]}...")
    logger.info("=" * 80)
    
    return state


def patch_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Execute Patch Generation Agent and update orchestrator state
    """
    logger.info("=" * 80)
    logger.info("EXECUTING PATCH GENERATION AGENT")
    logger.info("=" * 80)
    
    # Mark current agent
    state["current_agent"] = "Patch Generation Agent"
    state["workflow_status"] = "running_patch"
    
    # Prepare Patch agent input state (matches PatchState structure)
    patch_input = {
        "messages": [HumanMessage(content="Generate patch using Fix Plan and tools")],
        "shared_memory": {
            "rca_result": state["shared_memory"]["rca_result"],
            "fix_result": state["shared_memory"]["fix_result"],
            "patch_iteration": 0
        },
        "message_history": []
    }
    
    # Execute Patch agent
    patch_result = patch_app.invoke(patch_input)
    
    # Extract Patch output from agent's shared memory
    patch_output = patch_result["shared_memory"].get("patch_result")
    
    if not patch_output:
        raise ValueError("Patch Agent failed to produce output")
    
    # Update orchestrator state
    state["shared_memory"]["patch_result"] = patch_output
    state["shared_memory"]["patch_iteration"] = patch_result["shared_memory"].get("patch_iteration", 0)
    
    # Append Patch message history to global history
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
    
    # Add agent nodes
    graph.add_node("rca_agent", rca_agent_node)
    graph.add_node("fix_agent", fix_agent_node)
    graph.add_node("patch_agent", patch_agent_node)
    
    # Define sequential workflow: START -> RCA -> Fix -> Patch -> END
    graph.add_edge(START, "rca_agent")
    graph.add_edge("rca_agent", "fix_agent")
    graph.add_edge("fix_agent", "patch_agent")
    graph.add_edge("patch_agent", END)
    
    return graph.compile()


def main():
    """
    Main orchestration function that runs the complete multi-agent workflow
    """
    # Configuration
    CODEBASE_ROOT = os.getenv("CODEBASE_ROOT", r"D:\Siddhi\projects\RCA-Agent\codebase")
    TRACE_FILE = os.getenv("TRACE_FILE", r"D:\Siddhi\projects\RCA-Agent\trace_1.json")
    OUTPUT_DIR = Path("output")
    
    # Set environment variable for tools
    os.environ["CODEBASE_ROOT"] = CODEBASE_ROOT
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("MULTI-AGENT RCA SYSTEM STARTED")
    logger.info(f"Codebase Root: {CODEBASE_ROOT}")
    logger.info(f"Trace File: {TRACE_FILE}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info("=" * 80)
    
    # Load trace file as STRING (not JSON object)
    try:
        with open(TRACE_FILE, "r", encoding="utf-8") as f:
            trace_data = f.read()  # Read as string
        logger.info(f"Trace file loaded successfully - Size: {len(trace_data)} bytes")
        logger.info(f"Preview: {trace_data[:200]}...")
    except Exception as e:
        logger.error(f"Failed to load trace file: {str(e)}")
        return
    
    # Initialize orchestrator state
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
    
    # Execute the multi-agent workflow
    try:
        # Build orchestrator
        orchestrator = build_orchestration_graph()
        
        logger.info("=" * 80)
        logger.info("STARTING AGENT ORCHESTRATION")
        logger.info("=" * 80)
        
        # Execute the complete workflow
        final_state = orchestrator.invoke(initial_state)
        
        # Add workflow completion event
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
        
        # Add failure event
        initial_state["message_history"].append({
            "event": "workflow_failed",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        })
        final_state = initial_state
        final_state["workflow_status"] = "failed"
    
    # Save message history
    message_history_path = OUTPUT_DIR / "message_history.json"
    with open(message_history_path, "w", encoding="utf-8") as f:
        json.dump(final_state["message_history"], f, indent=2, ensure_ascii=False)
    logger.info(f"Message history saved to: {message_history_path}")
    
    # Save shared memory
    shared_memory_path = OUTPUT_DIR / "shared_memory.json"
    with open(shared_memory_path, "w", encoding="utf-8") as f:
        json.dump(final_state["shared_memory"], f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Shared memory saved to: {shared_memory_path}")
    
    # Print final summary
    logger.info("=" * 80)
    logger.info("WORKFLOW SUMMARY")
    logger.info(f"Status: {final_state['workflow_status']}")
    logger.info(f"Total Messages: {len(final_state['message_history'])}")
    logger.info(f"RCA Completed: {final_state['shared_memory'].get('rca_result') is not None}")
    logger.info(f"Fix Completed: {final_state['shared_memory'].get('fix_result') is not None}")
    logger.info(f"Patch Completed: {final_state['shared_memory'].get('patch_result') is not None}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info("=" * 80)
    
    # Display patch information if available
    if final_state["shared_memory"].get("patch_result"):
        patch_info = final_state["shared_memory"]["patch_result"]
        logger.info("=" * 80)
        logger.info("PATCH FILE GENERATED")
        logger.info(f"Success: {patch_info.get('success')}")
        logger.info(f"Patch File: {patch_info.get('patch_file')}")
        logger.info(f"Original File: {patch_info.get('original_file')}")
        logger.info("=" * 80)
    
    # Final output summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ Multi-Agent RCA System execution completed!")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    logger.info(f"📄 Message History: {message_history_path}")
    logger.info(f"📄 Shared Memory: {shared_memory_path}")
    logger.info(f"📄 Log File: orchestrator.log")
    
    if final_state["shared_memory"].get("patch_result"):
        patch_info = final_state["shared_memory"]["patch_result"]
        if patch_info.get("success"):
            logger.info(f"🔧 Patch File: {patch_info.get('patch_file')}")
        else:
            logger.info(f"❌ Patch Generation Failed: {patch_info.get('error', 'Unknown error')}")
    
    logger.info("=" * 80)
    logger.info("")


if __name__ == "__main__":
    main()