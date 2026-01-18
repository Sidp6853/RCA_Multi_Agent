import os
from pathlib import Path
from datetime import datetime
import json
from dotenv import load_dotenv
from typing import Dict, Any, Annotated
from pydantic import BaseModel, Field
from operator import add
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState 
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.exceptions import OutputParserException


from app.tools.read_file_tool import read_file
from app.tools.get_project_directory_tool import get_project_directory
from app.tools.check_dependency_tool import check_dependency
from app.prompts.rca import SYSTEM_PROMPT

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




def log_console(title: str, data=None):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    if data is not None:
        print(data)


class RCAState(MessagesState):
    shared_memory: Annotated[Dict[str, Any], lambda x, y: {**(x or {}), **y}]  
    message_history: Annotated[list, add]



class RCAOutput(BaseModel):
    error_type: str
    error_message: str
    root_cause: str
    affected_file: str
    affected_line: int
    



model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    streaming=False
)

tools = [read_file, get_project_directory, check_dependency]
model_with_tools = model.bind_tools(tools)


rca_parser = PydanticOutputParser(pydantic_object=RCAOutput)
fixing_parser = OutputFixingParser.from_llm(parser=rca_parser, llm=model)

def rca_llm_node(state: RCAState) -> Dict[str, Any]:
    iteration = state["shared_memory"].get("iteration", 0) + 1
    step_type = "INITIAL" if iteration == 1 else "REFINEMENT"
    
    agent_input = state["messages"][-1].content
    logger.info(f"[ITER {iteration}] AGENT INPUT ({step_type}): {agent_input}")

    history_entry = {
        "event": "agent_input",
        "agent": "RCA",
        "iteration": iteration,
        "content": agent_input
    }

    
    response = model_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    
    history_entry_out = {
        "event": "agent_output", 
        "agent": "RCA",
        "iteration": iteration,
        "content": response.text,
        "tool_calls": getattr(response, "tool_calls", [])
    }
    
    shared_update = {"iteration": iteration}
    if not getattr(response, "tool_calls", []):
        try:
            structured_rca = rca_parser.parse(response.content)
            shared_update["rca_result"] = structured_rca.model_dump()
        except OutputParserException:
            try: 

                structured_rca = fixing_parser.parse(response.content)
                shared_update["rca_result"] = structured_rca.model_dump()

            except Exception as e:
            
                logger.error(f"[ITER {iteration}] PARSING FAILED: {str(e)}")
                shared_update["rca_result"] = {
                    "error_type": "ParsingError",
                    "error_message": f"Failed to parse RCA output: {str(e)}",
                    "root_cause": response.content,
                    "affected_file": "unknown",
                    "affected_line": 0
                }    


    
    return {
        "messages": [response],
        "shared_memory": shared_update,
        "message_history": [history_entry, history_entry_out]
    }

def tool_node(state: RCAState) -> Dict[str, Any]:
    tool_results = []
    history_entries = []
    
    last_message = state["messages"][-1]
    for tool_call in getattr(last_message, "tool_calls", []):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name == "read_file":
            observation = read_file.invoke(tool_args)
        elif tool_name == "get_project_directory":
            observation = get_project_directory.invoke(tool_args)
        elif tool_name == "check_dependency":
            observation = check_dependency.invoke(tool_args)
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

def should_continue(state: RCAState):
    if state["shared_memory"].get("iteration", 0) >= 5:
        return END
    
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None): 
        return "tools"
    
    
    if state["shared_memory"].get("rca_result"):
        return END
        
    return "llm" 



graph = StateGraph(RCAState)
graph.add_node("llm", rca_llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", END])
graph.add_edge("tools", "llm")
rca_app = graph.compile()




