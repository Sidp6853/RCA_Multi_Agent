from datetime import datetime
from typing import Dict, Any, Annotated
from pydantic import BaseModel, Field
from operator import add
import logging
import time
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState 

from app.tools.read_file_tool import read_file
from app.tools.get_project_directory_tool import get_project_directory
from app.tools.check_dependency_tool import check_dependency
from app.prompts.rca import SYSTEM_PROMPT




logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RCAState(MessagesState):
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



#Output Schema
class RCAOutput(BaseModel):
    """Root Cause Analysis output"""
    error_type: str = Field(description="Type of error encountered")
    error_message: str = Field(description="The error message from the stacktrace")
    root_cause: str = Field(description="Detailed explanation of what caused the error")
    affected_file: str = Field(description="Path to the file where the error occurred")
    affected_line: int = Field(description="Line number where the error occurred")

#Toollist
tools = [read_file, get_project_directory, check_dependency]
#Tool Binding to the model
model_with_tools = base_model.bind_tools(tools)

#Structured Output 
model_with_structured_output = base_model.with_structured_output(
    RCAOutput,
    include_raw=True
)


def rca_llm_node(state: RCAState) -> Dict[str, Any]:
    iteration = state["shared_memory"].get("iteration", 0) + 1

    agent_input = state["messages"][-1].content

    logger.info(f"[ITER {iteration}] RCA INPUT: {agent_input}")

    history_entry = {
        "event": "agent_input",
        "agent": "RCA",
        "iteration": iteration,
        "content": agent_input
    }

    try:
        
        conversation = [SystemMessage(content=SYSTEM_PROMPT)]
        conversation.extend(state["messages"])

        time.sleep(15)
        
        
        result = model_with_tools.invoke(conversation)
        
        
        if getattr(result, "tool_calls", None):
            logger.info(f"[ITER {iteration}] Tool calls requested: {result.tool_calls}")
            return {
                "messages": [result],
                "shared_memory": {"iteration": iteration},
                "message_history": [history_entry]
            }
        
       
        time.sleep(15)
        structured_result = model_with_structured_output.invoke(conversation)
        
        
        if isinstance(structured_result, dict) and "parsed" in structured_result:
            parsed_output = structured_result["parsed"]
            raw_response = structured_result.get("raw", structured_result)
            logger.info(f"[ITER {iteration}] Structured output received successfully")
        else:
            
            parsed_output = structured_result
            raw_response = structured_result
        
        
        if not isinstance(parsed_output, RCAOutput):
            
            if isinstance(parsed_output, dict):
                parsed_output = RCAOutput(**parsed_output)
            else:
                raise ValueError(f"Unexpected output type: {type(parsed_output)}")
        
        rca_result = parsed_output.model_dump()
        
        logger.info(f"[ITER {iteration}] ✅ RCA RESULT: {rca_result}")

        history_entry_out = {
            "event": "agent_output",
            "agent": "RCA",
            "iteration": iteration,
            "rca_result": rca_result
        }

        shared_update = {
            "iteration": iteration,
            "rca_result": rca_result
        }

        return {
            "messages": [HumanMessage(content=f"RCA completed: {rca_result}")],
            "shared_memory": shared_update,
            "message_history": [history_entry, history_entry_out]
        }
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f" RCA FAILURE: {error_msg}")
        
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        history_entry_error = {
            "event": "rca_failed",
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "error": error_msg
        }
        
        shared_update = {
            "iteration": iteration,
            "rca_result": None,
            "error": error_msg
        }
        
        return {
            "messages": [],
            "shared_memory": shared_update,
            "message_history": [history_entry, history_entry_error]
        }


def tool_node(state: RCAState) -> Dict[str, Any]:
    tool_results = []
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
            time.sleep(30)
            observation = read_file.invoke(tool_args)
            
            if observation.get("success"):
                tool_content = f"""Successfully read file: {observation['file_path']}
Total lines: {observation.get('lines', 'N/A')}

File Content:
{observation['content']}"""
            else:
                tool_content = f"Error: {observation.get('error', 'Unknown error')}"
                
        elif tool_name == "get_project_directory":
            time.sleep(15)
            observation = get_project_directory.invoke(tool_args)
            tool_content = str(observation)
            
            
        elif tool_name == "check_dependency":
            time.sleep(15)
            observation = check_dependency.invoke(tool_args)

            tool_content = str(observation)
           
            
        else:
            observation = f"Unknown tool: {tool_name}"
            tool_content = observation

        logger.info(f"[TOOL RESULT] {tool_name}: {str(observation)}")
        
        history_entries.append({
            "event": "tool_call",
            "iteration": state["shared_memory"].get("iteration", 0),
            "tool": tool_name,
            "input": tool_args,
            "output": observation
        })
        
        tool_results.append(ToolMessage(
            content=tool_content,
            tool_call_id=tool_call["id"],
            name=tool_name
        ))
    
    return {
        "messages": tool_results,
        "message_history": history_entries
    }


def should_continue(state: RCAState):
    if state["shared_memory"].get("iteration", 0) >= 5:
        logger.warning(" Max iterations reached")
        return END
    
    last_message = state["messages"][-1]
    
    
    if getattr(last_message, "tool_calls", None):
        return "tools"
    
    
    if state["shared_memory"].get("rca_result"):
        logger.info("RCA completed successfully")
        return END
    
    
    return "llm"


#StateGraph
graph = StateGraph(RCAState)
graph.add_node("llm", rca_llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", "llm", END])
graph.add_edge("tools", "llm")
rca_app = graph.compile()   