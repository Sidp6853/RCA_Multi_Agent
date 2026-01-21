from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver 

from app.agents.RCA_agent import rca_app
from app.agents.fix_agent import fix_app
from app.agents.patch_agent import patch_app

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Pipeline state passing results between agents"""
    messages: List = field(default_factory=list)
    rca_result: Optional[Dict[str, Any]] = None
    fix_result: Optional[Dict[str, Any]] = None
    patch_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def rca_node(state: PipelineState) -> PipelineState:
    """Run RCA agent"""
    logger.info("Running RCA Agent")
    
    # Prepare input matching RCAState structure
    rca_input = {
        "messages": state.messages,
        "shared_memory": {"iteration": 0},
        "message_history": state.messages.copy()
    }
    
    rca_result = rca_app.invoke(rca_input)
    state.messages.extend(rca_result.get("messages", []))
    
    # Extract RCA result from shared_memory
    state.rca_result = rca_result["shared_memory"].get("rca_result")
    
    if not state.rca_result:
        state.error = "RCA failed to produce output"
        logger.error(state.error)
        return state
    
    logger.info(f"RCA: {state.rca_result.get('affected_file')}:{state.rca_result.get('affected_line')}")
    return state


def fix_node(state: PipelineState) -> PipelineState:
    """Run Fix agent"""
    if not state.rca_result:
        state.error = "No RCA result available for Fix agent"
        logger.error(state.error)
        return state
        
    logger.info("Running Fix Agent")
    
    # Prepare input matching FixState structure
    fix_input = {
        "messages": [HumanMessage(content="Generate fix plan based on RCA")],
        "shared_memory": {
            "rca_result": state.rca_result,
            "iteration": 0
        },
        "message_history": state.messages.copy()
    }
    
    fix_result = fix_app.invoke(fix_input)
    state.messages.extend(fix_result.get("messages", []))
    
    state.fix_result = fix_result["shared_memory"].get("fix_result")
    
    if not state.fix_result:
        state.error = "Fix agent failed to produce output"
        logger.error(state.error)
        return state
    
    logger.info(f"Fix: {state.fix_result.get('fix_summary')}")
    return state


def patch_node(state: PipelineState) -> PipelineState:
    """Run Patch agent""" 
    if not state.rca_result or not state.fix_result:
        state.error = "Missing RCA/Fix results for Patch agent"
        logger.error(state.error)
        return state
    
    logger.info("Running Patch Agent")
    
    # Prepare input matching PatchState structure
    patch_input = {
        "messages": [HumanMessage(content="Generate patch using Fix Plan and tools")],
        "shared_memory": {
            "rca_result": state.rca_result,
            "fix_result": state.fix_result,
            "patch_iteration": 0
        },
        "message_history": state.messages.copy()
    }
    
    patch_result = patch_app.invoke(patch_input)
    state.messages.extend(patch_result.get("messages", []))
    
    state.patch_result = patch_result["shared_memory"].get("patch_result")
    
    if state.patch_result and state.patch_result.get("success"):
        logger.info(f"Patch: {state.patch_result.get('patch_file')}")
    else:
        state.error = "Patch agent failed to create patch"
        logger.error(state.error)
    
    return state


# Build the workflow graph
workflow = StateGraph(PipelineState)
checkpointer = MemorySaver()

workflow.add_node("rca", rca_node)
workflow.add_node("fix", fix_node) 
workflow.add_node("patch", patch_node)

workflow.add_edge(START, "rca")
workflow.add_edge("rca", "fix")
workflow.add_edge("fix", "patch")
workflow.add_edge("patch", END)

pipeline = workflow.compile(checkpointer=checkpointer)