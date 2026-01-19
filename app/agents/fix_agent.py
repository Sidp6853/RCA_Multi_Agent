from typing import Dict, Any, List, Annotated
from pydantic import BaseModel
import time
import logging
from operator import add

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

from app.tools.read_file_tool import read_file
from app.prompts.fix import SYSTEM_PROMPT
from app.config.Model import ModelConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import os

load_dotenv()
base_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.getenv('GOOGLE_API_KEY'),
            temperature=0,
            streaming=False
        )



class FixState(MessagesState):
    shared_memory: Annotated[Dict[str, Any], lambda x, y: {**(x or {}), **y}]  
    message_history: Annotated[list, add]

#Output Schema
class FixOutput(BaseModel):
    fix_summary: str
    files_to_modify: List[str]
    patch_plan: List[str]
    safety_considerations: str

    class Config:
        extra = "forbid"
        

tools = [read_file]
#Tool Binding 
model_with_tools = base_model.bind_tools(tools)


model_with_structured_output = base_model.with_structured_output(
    FixOutput,
    include_raw=True
)

def fix_llm_node(state: FixState):
    iteration = state["shared_memory"].get("iteration", 0) + 1 
    step_type = "INITIAL" if iteration == 1 else "REFINEMENT"

    rca_result = state["shared_memory"].get("rca_result")
    if not rca_result:
        raise ValueError("RCA output missing in shared memory")

    agent_input = f"""
Root Cause Analysis:

Error Type: {rca_result['error_type']}
Error Message: {rca_result['error_message']}
Root Cause: {rca_result['root_cause']}
Affected File: {rca_result['affected_file']}
Affected Line: {rca_result['affected_line']}

You MUST use the Affected File and Affected Line provided in RCA.
"""

    logger.info(f"[ITER {iteration}] AGENT INPUT ({step_type})")

    history_entry = {
        "event": "agent_input",
        "agent": "Fix",
        "iteration": iteration,
        "content": agent_input
    }

    try:
        conversation = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=agent_input)
        ]

        time.sleep(15)
        
        #Check for tools first
        tool_check = model_with_tools.invoke(conversation)
        
        if getattr(tool_check, "tool_calls", None):
            logger.info(f"[ITER {iteration}] Tool calls requested")
            return {
                "messages": [tool_check],
                "shared_memory": {"iteration": iteration},
                "message_history": [history_entry]
            }
        
        # No tools → structured output
        time.sleep(15)
        result = model_with_structured_output.invoke(conversation)
        
        # Handle structured response
        if isinstance(result, dict) and "parsed" in result:
            structured_fix = result["parsed"]
            raw_response = result.get("raw", result)
        else:
            structured_fix = result
            raw_response = result
        
        # Force affected_file from RCA as Agent was hallucinating without this 
        fix_data = structured_fix.model_dump()
        fix_data["files_to_modify"] = [rca_result["affected_file"]]
        
        logger.info(f"[ITER {iteration}]  FIX PLAN: {fix_data['fix_summary']}")

        history_entry_out = {
            "event": "agent_output",
            "agent": "Fix",
            "iteration": iteration,
            "fix_result": fix_data
        }

        shared_update = {
            "iteration": iteration,
            "fix_result": fix_data
        }

        return {
            "messages": [raw_response],
            "shared_memory": shared_update,
            "message_history": [history_entry, history_entry_out]
        }
    
    except Exception as e:
        logger.error(f"[ITER {iteration}] FIX FAILED: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Fallback fix plan
        fallback_fix = {
            "fix_summary": "Analysis failed",
            "files_to_modify": [rca_result["affected_file"]],
            "patch_plan": ["Manual fix required"],
            "safety_considerations": str(e)
        }
        
        return {
            "messages": [],
            "shared_memory": {
                "iteration": iteration,
                "fix_result": fallback_fix
            },
            "message_history": [{
                "event": "fix_failed",
                "iteration": iteration,
                "error": str(e)
            }]
        }



def tool_node(state: FixState) -> Dict[str, Any]:
    tool_results = []
    history_entries = []
    
    last_message = state["messages"][-1]
    for tool_call in getattr(last_message, "tool_calls", []):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name == "read_file":
            time.sleep(15)
            observation = read_file.invoke(tool_args)
    
        else:
            observation = f"Unknown tool: {tool_name}"

        logger.info(
            f"[ITER {state['shared_memory'].get('iteration', 0)}] TOOL EXECUTED -> {tool_name} | "
            f"Input: {tool_args} | Output: {observation}"
        )
        
        history_entries.append({
            "event": "tool_call",
            "iteration": state["shared_memory"].get("iteration"),
            "tool": tool_name,
            "input": tool_args,
            "output": observation
        })
        
        tool_results.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"],
            name=tool_name
        ))
    
    return {
        "messages": tool_results,
        "message_history": history_entries
    }


def should_continue(state: FixState):
    if state["shared_memory"].get("iteration", 0) >= 3:
        return END
    
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    
    if state["shared_memory"].get("fix_result"):
        return END
        
    return "llm"

graph = StateGraph(FixState)
graph.add_node("llm", fix_llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", END])
graph.add_edge("tools", "llm")
fix_app = graph.compile()
