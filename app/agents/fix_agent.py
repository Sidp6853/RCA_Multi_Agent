import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.exceptions import OutputParserException 
import time
import logging
from operator import add
from typing import Annotated

from app.tools.read_file_tool import read_file
from app.prompts.fix import SYSTEM_PROMPT


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FixState(MessagesState):
    shared_memory: Annotated[Dict[str, Any], lambda x, y: {**(x or {}), **y}]  
    message_history: Annotated[list, add]

class FixOutput(BaseModel):
    fix_summary: str
    files_to_modify: List[str]
    patch_plan: List[str]
    safety_considerations: str


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    streaming=False
)


fix_parser = PydanticOutputParser(pydantic_object=FixOutput)
fixing_parser = OutputFixingParser.from_llm(parser=fix_parser, llm=model)  # retries parsing errors

tools = [read_file]
model_with_tools = model.bind_tools(tools)


def fix_llm_node(state: FixState):
    iteration = state["shared_memory"].get("iteration", 0) + 1 
    step_type = "INITIAL" if iteration == 1 else "REFINEMENT"

    rca_result = state["shared_memory"].get("rca_result")
    if not rca_result:
        raise ValueError("RCA output missing in shared memory")

    
    agent_input = f"""
Root Cause Analysis:
Here's the Root Cause Analysis

Error Type: {rca_result['error_type']}
Error Message: {rca_result['error_message']}
Root Cause: {rca_result['root_cause']}
Affected File: {rca_result['affected_file']}
Affected Line: {rca_result['affected_line']}

You MUST use the Affected File and Affected Line provided in RCA.
DO NOT introduce new files.
files_to_modify MUST contain ONLY the RCA affected file.


Generate Fix Plan:

"""

    logger.info(f"[ITER {iteration}] AGENT INPUT ({step_type}): {agent_input}")

    history_entry = {
        "event": "agent_input",
        "agent": "Fix",
        "iteration": iteration,
        "content": agent_input
    }

    time.sleep(15)
    response = model_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=agent_input)])

    history_entry_out = {
        "event": "agent_output",
        "agent": "Fix",
        "iteration": iteration,
        "content": response.text,
        "tool_calls": getattr(response, "tool_calls", [])
    }


    shared_update = {"iteration": iteration}

    if not getattr(response, "tool_calls", []):
        try:
            structured_fix = fix_parser.parse(response.text)
            fix_data = structured_fix.model_dump()
            fix_data["files_to_modify"] = [rca_result["affected_file"]]
            shared_update["fix_result"] = fix_data
        except OutputParserException:
            try:
                structured_fix = fixing_parser.parse(response.content)
                fix_data = structured_fix.model_dump()
                fix_data["files_to_modify"] = [rca_result["affected_file"]]
                shared_update["fix_result"] = fix_data
            except Exception as e:
                logger.error(f"[ITER {iteration}] PARSING FAILED: {str(e)}")
                shared_update["fix_result"] = {
                    "fix_summary": "Failed to parse fix output",
                    "files_to_modify": [rca_result["affected_file"]],
                    "patch_plan": [],
                    "safety_considerations": f"Parsing error: {str(e)}"
                }




    return {
        "messages": [response],
        "shared_memory": shared_update,
        "message_history": [history_entry, history_entry_out]
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
            tool_call_id=tool_call["id"]
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
