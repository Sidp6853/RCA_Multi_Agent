from datetime import datetime
from typing import Dict, Any, Annotated
from operator import add
import time
from pydantic import BaseModel, Field
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

from app.tools.create_patch_tool import create_patch_file
from app.tools.read_file_tool import read_file
from app.tools.check_dependency_tool import check_dependency
from app.prompts.patch import SYSTEM_PROMPT
from app.config.Model import ModelConfig


import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PatchOutput(BaseModel):
    """Complete fixed Python file output"""
    patched_code: str = Field(
        description="Full fixed Python file content. No markdown formatting. No explanations. Just the complete corrected code."
    )

#Output Schema 
class PatchState(MessagesState):
    shared_memory: Annotated[Dict[str, Any], lambda x, y: {**(x or {}), **y}]
    message_history: Annotated[list, add]

from dotenv import load_dotenv
import os

load_dotenv()
base_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.getenv('GOOGLE_API_KEY'),
            temperature=0,
            streaming=False
        )
 


tools = [read_file, check_dependency, create_patch_file]
model_with_tools = base_model.bind_tools(tools)


model_with_structured_output = base_model.with_structured_output(
    PatchOutput,
    include_raw=True  # Included raw response for debugging
)


def patch_llm_node(state: PatchState):
    iteration = state["shared_memory"].get("iteration", 0) + 1
    step_type = "INITIAL" if iteration == 1 else "REFINEMENT"

    rca_result = state["shared_memory"].get("rca_result", {})
    fix_result = state["shared_memory"].get("fix_result", {})

    if not fix_result:
        raise ValueError("Fix plan missing in shared memory")

    files_to_modify = fix_result.get('files_to_modify', [])
    if not files_to_modify:
        raise ValueError("No files to modify in fix plan")

    target_file = files_to_modify[0]

    # To check if we already have the file content from a previous tool call
    has_file_content = False
    for msg in reversed(state["messages"]):
        if hasattr(msg, 'name') and msg.name == "read_file":
            if "File Content:" in msg.content:
                has_file_content = True
                break

    if not has_file_content:
        # request file read 
        fix_text = f"""
ROOT CAUSE ANALYSIS RESULTS:

Error Type: {rca_result['error_type']}
Error Message: {rca_result['error_message']}
Affected File: {rca_result['affected_file']}
Affected Line: {rca_result['affected_line']}

Root Cause: {rca_result['root_cause']}

FIX PLAN:

Fix Summary: {fix_result['fix_summary']}
Files To Modify: {', '.join(files_to_modify)}

Step-by-Step Patch Plan:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(fix_result['patch_plan']))}

YOUR TASK:
Call the read_file tool with file_path="{target_file}" to get the complete original file content.
"""
        logger.info(f"[ITER {iteration}] REQUESTING FILE READ")
        
        conversation = [SystemMessage(content=SYSTEM_PROMPT)]
        conversation.extend(state["messages"])
        conversation.append(HumanMessage(content=fix_text))

        time.sleep(15)
        response = model_with_tools.invoke(conversation)
        
        return {
            "messages": [response],
            "shared_memory": {"patch_iteration": iteration},
            "message_history": [{
                "event": "tool_request",
                "agent": "PatchAgent",
                "iteration": iteration,
                "content": fix_text
            }]
        }

    
    fix_text = f"""
ROOT CAUSE ANALYSIS RESULTS:

Error Type: {rca_result['error_type']}
Error Message: {rca_result['error_message']}
Affected File: {rca_result['affected_file']}
Affected Line: {rca_result['affected_line']}

Root Cause: {rca_result['root_cause']}

FIX PLAN:

Fix Summary: {fix_result['fix_summary']}
Step-by-Step Patch Plan:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(fix_result['patch_plan']))}

Safety Considerations: {fix_result['safety_considerations']}

YOUR TASK:
Based on the file content you received, generate the COMPLETE fixed file with ONLY line {rca_result['affected_line']} changed according to the fix plan.

Output the complete fixed Python file. Include ALL the original code with just the one line fixed.
"""

    logger.info(f"[ITER {iteration}] GENERATING PATCH ({step_type})")

    history_entry = {
        "event": "patch_generation",
        "agent": "PatchAgent",
        "iteration": iteration,
        "content": fix_text
    }

    try:
        conversation = [SystemMessage(content=SYSTEM_PROMPT)]
        conversation.extend(state["messages"])
        conversation.append(HumanMessage(content=fix_text))

        time.sleep(15)
        
        
        result = model_with_structured_output.invoke(conversation)
        
        
        if isinstance(result, dict) and "parsed" in result:
            structured_patch = result["parsed"]
            raw_output = result.get("raw", {})
            logger.info(f"[ITER {iteration}] Structured output received successfully")
        else:
            
            structured_patch = result
            raw_output = result

        
        if isinstance(structured_patch, PatchOutput):
            patched_code = structured_patch.patched_code.strip()
        elif isinstance(structured_patch, dict):
            patched_code = structured_patch.get("patched_code", "").strip()
        else:
            raise ValueError(f"Unexpected structured_patch type: {type(structured_patch)}")

        if not patched_code:
            raise ValueError("Empty patched_code received from model")

        logger.info(f"[ITER {iteration}] Patched code length: {len(patched_code)} characters")

    
        patch_result = {}
        if files_to_modify and patched_code:
            original_file = files_to_modify[0]
            try:
                time.sleep(15)
                patch_result = create_patch_file.invoke({
                    "original_file_path": original_file,
                    "fixed_content": patched_code
                })
                logger.info(f"[ITER {iteration}]  PATCH CREATED: {patch_result}")
            except Exception as e:
                error_msg = f"Error creating patch file: {str(e)}"
                logger.error(f"PATCH CREATION ERROR: {error_msg}")
                patch_result = {"success": False, "error": error_msg}
        else:
            patch_result = {
                "success": False,
                "error": "No files to modify or empty patch code"
            }

        history_entry_out = {
            "event": "patch_created",
            "agent": "PatchAgent",
            "iteration": iteration,
            "patch_result": patch_result
        }

        shared_update = {
            "patch_iteration": iteration,
            "patch_result": patch_result
        }

        
        completion_message = HumanMessage(content=f"Patch generation completed. Result: {patch_result}")

        return {
            "messages": [completion_message],
            "shared_memory": shared_update,
            "message_history": [history_entry, history_entry_out]
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"PATCH FAILURE: {error_msg}")
        
        
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")

        history_entry_error = {
            "event": "patch_generation_failed",
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "error": error_msg
        }

        shared_update = {
            "patch_iteration": iteration,
            "patch_result": {
                "success": False,
                "error": error_msg
            }
        }

        return {
            "messages": [],
            "shared_memory": shared_update,
            "message_history": [history_entry, history_entry_error]
        }


def patch_tool_node(state: PatchState):
    results = []
    history_entries = []
    last_message = state["messages"][-1]

    if not getattr(last_message, "tool_calls", None):
        return {
            "messages": [],
            "message_history": []
        }

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        logger.info(f"[TOOL EXECUTION] {tool_name} | Input: {tool_args}")

        if tool_name == "read_file":
            time.sleep(15)
            observation = read_file.invoke(tool_args)
            
            if observation.get("success"):
                tool_content = f"""Successfully read file: {observation['file_path']}
Total lines: {observation.get('lines', 'N/A')}

File Content:
{observation['content']}"""
            else:
                tool_content = f"Error: {observation.get('error', 'Unknown error')}"

        elif tool_name == "check_dependency":
            time.sleep(15)
            observation = check_dependency.invoke(tool_args)
            tool_content = str(observation)

        elif tool_name == "create_patch_file":
            time.sleep(15)
            observation = create_patch_file.invoke(tool_args)
            tool_content = json.dumps(observation, indent=2)
            
            if observation.get("success"):
                logger.info(f"PATCH CREATED VIA TOOL CALL: {observation}")
            else:
                logger.error(f" PATCH CREATION FAILED: {observation}")

        else:
            observation = f"Unknown tool: {tool_name}"
            tool_content = observation

        history_entries.append({
            "event": "tool_call",
            "iteration": state["shared_memory"].get("iteration", 0),
            "tool": tool_name,
            "input": tool_args,
            "output": observation
        })

        logger.info(f"[TOOL RESULT] {tool_name}: {str(observation)[:500]}")

        results.append(
            ToolMessage(
                content=tool_content,
                name=tool_name,
                tool_call_id=tool_call.get("id")
            )
        )

    return {
        "messages": results,
        "message_history": history_entries
    }


def should_continue(state: PatchState):
    if state["shared_memory"].get("iteration", 0) >= 5:
        logger.warning("Max iterations reached")
        return END
    
    last_message = state["messages"][-1]
    
    # If the last message has tool calls, go to tools node
    if getattr(last_message, "tool_calls", None):
        return "tools"
    
    # If we have a successful patch result, end
    patch_result = state["shared_memory"].get("patch_result")
    if patch_result and patch_result.get("success"):
        logger.info("✅ Patch created successfully, ending workflow")
        return END
    
    # Check if we have file content from tools
    has_file_content = False
    for msg in reversed(state["messages"]):
        if hasattr(msg, 'name') and msg.name == "read_file":
            if "File Content:" in msg.content:
                has_file_content = True
                break
    
    # If we have file content but no patch yet, continue to LLM
    if has_file_content and not patch_result:
        return "llm"
    
    # Default: continue to LLM
    return "llm"


graph = StateGraph(PatchState)
graph.add_node("llm", patch_llm_node)
graph.add_node("tools", patch_tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", "llm", END])
graph.add_edge("tools", "llm")
patch_app = graph.compile()