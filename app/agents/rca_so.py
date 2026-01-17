import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

from app.tools.read_file_tool import read_file
from app.tools.get_project_directory_tool import get_project_directory
from app.tools.check_dependency_tool import check_dependency

load_dotenv()


# ---------------- CONSOLE LOGGER ----------------
def log_console(title: str, data=None):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    if data is not None:
        print(data)


# ---------------- STATE ----------------
class RCAState(MessagesState):
    shared_memory: Dict[str, Any]
    message_history: list


# ---------------- RCA OUTPUT SCHEMA ----------------
class RCAOutput(BaseModel):
    """Root Cause Analysis structured output"""
    error_type: str = Field(description="Type of error (e.g., AttributeError, TypeError)")
    error_message: str = Field(description="Brief summary of the error")
    root_cause: str = Field(description="Detailed explanation of why the error occurred")
    affected_file: str = Field(description="File path where error occurred")
    affected_line: int = Field(description="Line number of the error")
    fix_recommendation: str = Field(description="Specific fix recommendation with code example")


# ---------------- MODEL ----------------
# Base model for tool calling
base_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

# Model with tools
tools = [read_file, get_project_directory, check_dependency]
model_with_tools = base_model.bind_tools(tools)

# Model for structured output (used at the end)
structured_output_model = base_model.with_structured_output(RCAOutput)


# ---------------- PROMPT ----------------
SYSTEM_PROMPT = """You are an expert Root Cause Analysis (RCA) Agent specializing in debugging and error analysis for APM codebases.

YOUR MISSION:
Perform precise Root Cause Analysis using REAL codebase files accessed through tools.

AVAILABLE TOOLS:
1. get_project_directory: Map codebase structure, identify file locations
2. read_file: Read actual source code from specific files
3. check_dependency: Verify installed packages, versions, and dependency conflicts

STRICT RULES:
1. ALWAYS use tools to gather information - NEVER guess or assume
2. NEVER rely solely on stacktrace snippets - they may be incomplete
3. ALWAYS verify your analysis against actual source code
4. IGNORE framework/library paths (e.g., /usr/local/lib/, site-packages/)
5. Only analyze files you have successfully accessed with tools

REQUIRED PROCESS:

Step 1: UNDERSTAND PROJECT STRUCTURE
   - Use get_project_directory to explore the codebase layout
   - Identify key directories (app/, src/, lib/, etc.)
   - Locate where the error files might be located

Step 2: PARSE ERROR TRACE
   - Identify all application files in the stack trace
   - Note the exact error location (file path + line number)
   - Filter out external library paths

Step 3: CHECK DEPENDENCIES (if relevant)
   - If error involves ImportError, ModuleNotFoundError, or AttributeError on imported modules
   - Use check_dependency to verify if required packages are installed

Step 4: IDENTIFY ROOT FILE
   - Find the FIRST application file where the error originated
   - Extract the relative path (remove /usr/srv/app prefix)

Step 5: READ SOURCE FILE
   - Use read_file tool with the relative path
   - Verify the file was read successfully
   - Read related files if needed

Step 6: ANALYZE ACTUAL CODE
   - Examine the EXACT line mentioned in the error
   - Look at surrounding context
   - Identify what the code is trying to do
   - Compare against the error message

Step 7: IDENTIFY ROOT CAUSE
   - What exactly went wrong?
   - Why did it fail?
   - What was expected vs what actually happened?

After gathering all information through tools, provide your analysis."""

FINAL_ANALYSIS_PROMPT = """Based on all the information you gathered through tools, provide your complete Root Cause Analysis.

You must respond with a structured output containing:
- error_type: Type of error (e.g., AttributeError)
- error_message: Brief summary of what failed
- root_cause: Detailed explanation of why it happened with line references
- affected_file: The file path where the error occurred
- affected_line: The line number
- fix_recommendation: Specific code change needed to fix the issue

Provide your final analysis now."""


# ---------------- LLM NODE ----------------
def rca_llm_node(state: RCAState):
    iteration = state["shared_memory"]["iteration"] + 1
    state["shared_memory"]["iteration"] = iteration
    step_type = "INITIAL" if iteration == 1 else "REFINEMENT"

    agent_input = state["messages"][-1].content

    state["message_history"].append({
        "event": "agent_input",
        "agent": "RCA",
        "iteration": iteration,
        "content": agent_input
    })
    log_console(f"[ITER {iteration}] AGENT INPUT ({step_type})", agent_input)

    # Check if we should generate final structured output
    if state["shared_memory"].get("ready_for_structured_output"):
        log_console(f"[ITER {iteration}] GENERATING STRUCTURED OUTPUT", "Using with_structured_output()")
        
        # Use structured output model
        structured_response = structured_output_model.invoke(
            [SystemMessage(content=SYSTEM_PROMPT + "\n\n" + FINAL_ANALYSIS_PROMPT)] + state["messages"]
        )
        
        state["shared_memory"]["rca_result"] = structured_response.model_dump()
        
        log_console(f"[ITER {iteration}] STRUCTURED OUTPUT", structured_response.model_dump())
        
        return {
            "messages": [],
            "shared_memory": state["shared_memory"],
            "message_history": state["message_history"]
        }

    # Regular tool-calling flow
    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )

    state["message_history"].append({
        "event": "agent_output",
        "agent": "RCA",
        "iteration": iteration,
        "content": response.content,
        "tool_calls": response.tool_calls if hasattr(response, 'tool_calls') else []
    })
    
    log_console(f"[ITER {iteration}] AGENT OUTPUT ({step_type})", response.content)
    if hasattr(response, 'tool_calls') and response.tool_calls:
        log_console(f"[ITER {iteration}] TOOL REQUESTS", response.tool_calls)

    return {
        "messages": [response],
        "shared_memory": state["shared_memory"],
        "message_history": state["message_history"]
    }


# ---------------- TOOL NODE ----------------
def tool_node(state: RCAState):
    results = []
    last_message = state["messages"][-1]

    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {
            "messages": [],
            "shared_memory": state["shared_memory"],
            "message_history": state["message_history"]
        }

    for tool_call in last_message.tool_calls:
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

        state["message_history"].append({
            "event": "tool_call",
            "iteration": state["shared_memory"]["iteration"],
            "tool": tool_name,
            "input": tool_args,
            "output": observation
        })

        log_console(
            f"[ITER {state['shared_memory']['iteration']}] TOOL EXECUTED -> {tool_name}",
            {"input": tool_args, "output": observation}
        )

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )

    return {
        "messages": results,
        "shared_memory": state["shared_memory"],
        "message_history": state["message_history"]
    }


# ---------------- FLOW CONTROL ----------------
def should_continue(state: RCAState):
    # If structured output generated, end
    if state["shared_memory"].get("rca_result"):
        return END
    
    # After 4 iterations, generate structured output
    if state["shared_memory"]["iteration"] >= 4:
        state["shared_memory"]["ready_for_structured_output"] = True
        return "llm"

    last_message = state["messages"][-1] if state["messages"] else None

    # If there are tool calls, continue to tools
    if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # No more tool calls, generate structured output
    state["shared_memory"]["ready_for_structured_output"] = True
    return "llm"


# ---------------- GRAPH ----------------
graph = StateGraph(RCAState)
graph.add_node("llm", rca_llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", "llm", END])
graph.add_edge("tools", "llm")
rca_app = graph.compile()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    os.environ["CODEBASE_ROOT"] = r"D:\Siddhi\projects\RCA-Agent\codebase"
    trace_path = r"D:\Siddhi\projects\RCA-Agent\trace_1.json"

    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = f.read()

    result = rca_app.invoke({
        "messages": [HumanMessage(content=trace_data)],
        "shared_memory": {
            "iteration": 0,
            "rca_result": None,
            "ready_for_structured_output": False
        },
        "message_history": []
    })

    final_output = result["shared_memory"]["rca_result"]

    result["message_history"].append({
        "event": "final_result",
        "iteration": result["shared_memory"]["iteration"],
        "content": final_output
    })
    log_console("FINAL RCA OUTPUT", final_output)

    # Save JSON files
    os.makedirs("langgraph_rca_system/output", exist_ok=True)
    with open("langgraph_rca_system/output/message_history.json", "w", encoding="utf-8") as f:
        json.dump(result["message_history"], f, indent=2)
    with open("langgraph_rca_system/output/shared_memory.json", "w", encoding="utf-8") as f:
        json.dump(result["shared_memory"], f, indent=2)

    log_console("FILES GENERATED", {
        "message_history.json": "Saved",
        "shared_memory.json": "Saved"
    })