import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, TypedDict, Annotated
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

# Import the individual agents
from app.agents.rca_agent import rca_app, RCAState
from app.agents.fix_agent import fix_app, FixState
from app.agents.patch_agent import patch_app, PatchState

load_dotenv()


# ============================================================================
# GLOBAL STATE FOR ORCHESTRATION
# ============================================================================
class OrchestratorState(TypedDict):
    """
    Global state that flows through the entire multi-agent workflow.
    This consolidates all agent states into a single unified state.
    """
    # Input
    trace_data: str
    codebase_root: str
    
    # Shared memory (accumulates results from all agents)
    shared_memory: Dict[str, Any]
    
    # Message history (complete log of all agent interactions)
    message_history: list
    
    # Workflow metadata
    workflow_status: str
    current_agent: str


# ============================================================================
# CONSOLE LOGGER
# ============================================================================
def log_console(title: str, data=None):
    """Enhanced logger with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 80)
    print(f"[{timestamp}] {title}")
    print("=" * 80)
    if data is not None:
        if isinstance(data, (dict, list)):
            print(json.dumps(data, indent=2))
        else:
            print(data)


# ============================================================================
# AGENT WRAPPER NODES
# ============================================================================

def rca_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Execute RCA Agent and update orchestrator state
    """
    log_console("EXECUTING RCA AGENT", {"trace_length": len(state["trace_data"])})
    
    # Mark current agent
    state["current_agent"] = "RCA Agent"
    state["workflow_status"] = "running_rca"
    
    # Prepare RCA agent input state
    rca_input = {
        "messages": [HumanMessage(content=state["trace_data"])],
        "shared_memory": {
            "iteration": 0,
            "rca_result": None
        },
        "message_history": []
    }
    
    # Execute RCA agent
    rca_result = rca_app.invoke(rca_input)
    
    # Extract results
    rca_output = rca_result["shared_memory"].get("rca_result")
    
    if not rca_output:
        raise ValueError("RCA Agent failed to produce output")
    
    # Update orchestrator state
    state["shared_memory"]["rca_result"] = rca_output
    state["shared_memory"]["rca_iteration"] = rca_result["shared_memory"]["iteration"]
    
    # Append RCA message history to global history
    for msg in rca_result["message_history"]:
        msg["agent"] = "RCA Agent"
        state["message_history"].append(msg)
    
    log_console("RCA AGENT COMPLETED", {
        "iterations": rca_result["shared_memory"]["iteration"],
        "error_type": rca_output.get("error_type"),
        "affected_file": rca_output.get("affected_file"),
        "affected_line": rca_output.get("affected_line")
    })
    
    return state


def fix_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Execute Fix Suggestion Agent and update orchestrator state
    """
    log_console("EXECUTING FIX SUGGESTION AGENT")
    
    # Mark current agent
    state["current_agent"] = "Fix Suggestion Agent"
    state["workflow_status"] = "running_fix"
    
    # Prepare Fix agent input state
    fix_input = {
        "messages": [],
        "shared_memory": {
            "rca_result": state["shared_memory"]["rca_result"],
            "fix_iteration": 0,
            "fix_result": None
        },
        "message_history": []
    }
    
    # Execute Fix agent
    fix_result = fix_app.invoke(fix_input)
    
    # Extract results
    fix_output = fix_result["shared_memory"].get("fix_result")
    
    if not fix_output:
        raise ValueError("Fix Agent failed to produce output")
    
    # Update orchestrator state
    state["shared_memory"]["fix_result"] = fix_output
    state["shared_memory"]["fix_iteration"] = fix_result["shared_memory"]["fix_iteration"]
    
    # Append Fix message history to global history
    for msg in fix_result["message_history"]:
        msg["agent"] = "Fix Suggestion Agent"
        state["message_history"].append(msg)
    
    log_console("FIX SUGGESTION AGENT COMPLETED", {
        "iterations": fix_result["shared_memory"]["fix_iteration"],
        "files_to_modify": fix_output.get("files_to_modify"),
        "patch_plan_steps": len(fix_output.get("patch_plan", []))
    })
    
    return state


def patch_agent_node(state: OrchestratorState) -> OrchestratorState:
    """
    Execute Patch Generation Agent and update orchestrator state
    """
    log_console("EXECUTING PATCH GENERATION AGENT")
    
    # Mark current agent
    state["current_agent"] = "Patch Generation Agent"
    state["workflow_status"] = "running_patch"
    
    # Prepare Patch agent input state
    patch_input = {
        "messages": [HumanMessage(content="Generate patch using Fix Plan and tools")],
        "shared_memory": {
            "rca_result": state["shared_memory"]["rca_result"],
            "fix_result": state["shared_memory"]["fix_result"],
            "patch_iteration": 0,
            "patch_result": None
        },
        "message_history": []
    }
    
    # Execute Patch agent
    patch_result = patch_app.invoke(patch_input)
    
    # Extract results
    patch_output = patch_result["shared_memory"].get("patch_result")
    
    if not patch_output:
        raise ValueError("Patch Agent failed to produce output")
    
    # Update orchestrator state
    state["shared_memory"]["patch_result"] = patch_output
    state["shared_memory"]["patch_iteration"] = patch_result["shared_memory"]["patch_iteration"]
    
    # Append Patch message history to global history
    for msg in patch_result["message_history"]:
        msg["agent"] = "Patch Generation Agent"
        state["message_history"].append(msg)
    
    log_console("PATCH GENERATION AGENT COMPLETED", {
        "iterations": patch_result["shared_memory"]["patch_iteration"],
        "patch_file_created": patch_output.get("success", False),
        "patch_file_path": patch_output.get("patch_file")
    })
    
    state["workflow_status"] = "completed"
    
    return state


# ============================================================================
# ORCHESTRATION GRAPH
# ============================================================================

def build_orchestration_graph():
    """
    Build the complete multi-agent orchestration graph
    
    Flow: START → RCA Agent → Fix Agent → Patch Agent → END
    """
    graph = StateGraph(OrchestratorState)
    
    # Add agent nodes
    graph.add_node("rca_agent", rca_agent_node)
    graph.add_node("fix_agent", fix_agent_node)
    graph.add_node("patch_agent", patch_agent_node)
    
    # Define linear workflow
    graph.add_edge(START, "rca_agent")
    graph.add_edge("rca_agent", "fix_agent")
    graph.add_edge("fix_agent", "patch_agent")
    graph.add_edge("patch_agent", END)
    
    return graph.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main entry point for the multi-agent RCA system
    """
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    CODEBASE_ROOT = os.getenv("CODEBASE_ROOT", r"D:\Siddhi\projects\RCA-Agent\codebase")
    TRACE_FILE = os.getenv("TRACE_FILE", r"D:\Siddhi\projects\RCA-Agent\trace_1.json")
    OUTPUT_DIR = Path("D:\Siddhi\projects\RCA-Agent\patches")
    
    # Set environment variable for tools
    os.environ["CODEBASE_ROOT"] = CODEBASE_ROOT
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    log_console("MULTI-AGENT RCA SYSTEM STARTED", {
        "codebase_root": CODEBASE_ROOT,
        "trace_file": TRACE_FILE,
        "output_dir": str(OUTPUT_DIR)
    })
    
    # ========================================================================
    # LOAD TRACE DATA
    # ========================================================================
    try:
        with open(TRACE_FILE, "r", encoding="utf-8") as f:
            trace_data = f.read()
        log_console("TRACE FILE LOADED", {"size": len(trace_data)})
    except Exception as e:
        log_console("ERROR LOADING TRACE FILE", str(e))
        return
    
    # ========================================================================
    # INITIALIZE STATE
    # ========================================================================
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
    
    # ========================================================================
    # BUILD AND EXECUTE ORCHESTRATION GRAPH
    # ========================================================================
    try:
        orchestrator = build_orchestration_graph()
        
        log_console("STARTING AGENT ORCHESTRATION")
        
        # Execute the complete workflow
        final_state = orchestrator.invoke(initial_state)
        
        # Add workflow completion event
        final_state["message_history"].append({
            "event": "workflow_completed",
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        })
        
        log_console("AGENT ORCHESTRATION COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        log_console("ORCHESTRATION FAILED", {
            "error": str(e),
            "error_type": type(e).__name__
        })
        
        # Add failure event
        initial_state["message_history"].append({
            "event": "workflow_failed",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "error_type": type(e).__name__
        })
        final_state = initial_state
        final_state["workflow_status"] = "failed"
    
    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================
    
    # 1. Save complete message history
    message_history_path = OUTPUT_DIR / "message_history.json"
    with open(message_history_path, "w", encoding="utf-8") as f:
        json.dump(final_state["message_history"], f, indent=2, ensure_ascii=False)
    log_console("MESSAGE HISTORY SAVED", str(message_history_path))
    
    # 2. Save final shared memory
    shared_memory_path = OUTPUT_DIR / "shared_memory.json"
    with open(shared_memory_path, "w", encoding="utf-8") as f:
        json.dump(final_state["shared_memory"], f, indent=2, ensure_ascii=False)
    log_console("SHARED MEMORY SAVED", str(shared_memory_path))
    
    # 3. Save individual agent outputs for easy reference
    
    # RCA Output
    if final_state["shared_memory"].get("rca_result"):
        rca_output_path = OUTPUT_DIR / "rca_output.json"
        with open(rca_output_path, "w", encoding="utf-8") as f:
            json.dump(final_state["shared_memory"]["rca_result"], f, indent=2, ensure_ascii=False)
        log_console("RCA OUTPUT SAVED", str(rca_output_path))
    
    # Fix Output
    if final_state["shared_memory"].get("fix_result"):
        fix_output_path = OUTPUT_DIR / "fix_output.json"
        with open(fix_output_path, "w", encoding="utf-8") as f:
            json.dump(final_state["shared_memory"]["fix_result"], f, indent=2, ensure_ascii=False)
        log_console("FIX OUTPUT SAVED", str(fix_output_path))
    
    # Patch Output
    if final_state["shared_memory"].get("patch_result"):
        patch_output_path = OUTPUT_DIR / "patch_output.json"
        with open(patch_output_path, "w", encoding="utf-8") as f:
            json.dump(final_state["shared_memory"]["patch_result"], f, indent=2, ensure_ascii=False)
        log_console("PATCH OUTPUT SAVED", str(patch_output_path))
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    log_console("=" * 80)
    log_console("WORKFLOW SUMMARY", {
        "status": final_state["workflow_status"],
        "total_messages": len(final_state["message_history"]),
        "rca_completed": final_state["shared_memory"].get("rca_result") is not None,
        "fix_completed": final_state["shared_memory"].get("fix_result") is not None,
        "patch_completed": final_state["shared_memory"].get("patch_result") is not None,
        "output_directory": str(OUTPUT_DIR)
    })
    
    if final_state["shared_memory"].get("patch_result"):
        patch_info = final_state["shared_memory"]["patch_result"]
        log_console("PATCH FILE GENERATED", {
            "success": patch_info.get("success"),
            "patch_file": patch_info.get("patch_file"),
            "original_file": patch_info.get("original_file")
        })
    
    log_console("=" * 80)
    print("\n✅ Multi-Agent RCA System execution completed!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📄 Message History: {message_history_path}")
    print(f"📄 Shared Memory: {shared_memory_path}")
    if final_state["shared_memory"].get("patch_result"):
        print(f"🔧 Patch File: {patch_info.get('patch_file')}")
    print()


if __name__ == "__main__":
    main()