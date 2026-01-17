import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, List
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.exceptions import OutputParserException

from app.tools.read_file_tool import read_file
from app.tools.get_project_directory_tool import get_project_directory

load_dotenv()

# ---------------- CONSOLE LOGGER ----------------
def log_console(title: str, data=None):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    if data is not None:
        print(data)

# ---------------- STATE ----------------
class FixState(MessagesState):
    shared_memory: Dict[str, Any]
    message_history: list

# ---------------- FIX OUTPUT SCHEMA ----------------
class FixOutput(BaseModel):
    fix_summary: str
    files_to_modify: List[str]
    patch_plan: List[str]
    safety_considerations: str

# ---------------- MODEL ----------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    streaming=False
)

fix_parser = PydanticOutputParser(pydantic_object=FixOutput)
fixing_parser = OutputFixingParser.from_llm(parser=fix_parser, llm=model)  # retries parsing errors

# ---------------- PROMPT ----------------
SYSTEM_PROMPT = """
You are a Fix Suggestion Agent.

Responsibilities:
1. Read the Root Cause Analysis (RCA) output
2. Generate a clear fix plan
3. Include safety checks
4. Describe exactly what code changes are required
5. DO NOT implement code — only generate instructions

STRICT RULES:
- Base decisions ONLY on RCA output
- Use tools to verify files if needed
- Do not guess
- Do not hallucinate fixes

OUTPUT FORMAT:
{
  "fix_summary": "...",
  "files_to_modify": ["..."],
  "patch_plan": ["step-by-step instructions"],
  "safety_considerations": "..."
}
"""

# ---------------- TOOLS ----------------
tools = [read_file, get_project_directory]
model_with_tools = model.bind_tools(tools)

# ---------------- LLM NODE ----------------
def fix_llm_node(state: FixState):
    iteration = state["shared_memory"].get("fix_iteration", 0) + 1
    state["shared_memory"]["fix_iteration"] = iteration

    rca_result = state["shared_memory"].get("rca_result")
    if not rca_result:
        raise ValueError("RCA output missing in shared memory")

    # Format RCA into structured prompt
    agent_input = f"""
Root Cause Analysis:

Error Type: {rca_result['error_type']}
Error Message: {rca_result['error_message']}
Root Cause: {rca_result['root_cause']}
Affected File: {rca_result['affected_file']}
Affected Line: {rca_result['affected_line']}

Generate Fix Plan:
"""

    # Log agent input
    state["message_history"].append({
        "event": "agent_input",
        "agent": "Fix",
        "iteration": iteration,
        "content": agent_input
    })
    log_console(f"[ITER {iteration}] AGENT INPUT", agent_input)

    # Invoke model
    response = model_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=agent_input)])

    # Log agent output
    state["message_history"].append({
        "event": "agent_output",
        "agent": "Fix",
        "iteration": iteration,
        "content": response.content,
        "tool_calls": response.tool_calls
    })
    log_console(f"[ITER {iteration}] AGENT OUTPUT", response.content)
    if response.tool_calls:
        log_console(f"[ITER {iteration}] TOOL REQUESTS", response.tool_calls)

    # Parse structured Fix output
    try:
        structured_fix = fix_parser.parse(response.content)
    except OutputParserException:
        structured_fix = fixing_parser.parse(response.content)

    state["shared_memory"]["fix_result"] = structured_fix.model_dump()

    return {
        "messages": [response],
        "shared_memory": state["shared_memory"],
        "message_history": state["message_history"]
    }

# ---------------- TOOL NODE ----------------
def tool_node(state: FixState):
    results = []
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
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
        else:
            observation = f"Unknown tool: {tool_name}"

        state["message_history"].append({
            "event": "tool_call",
            "iteration": state["shared_memory"]["fix_iteration"],
            "tool": tool_name,
            "input": tool_args,
            "output": observation
        })

        log_console(
            f"[ITER {state['shared_memory']['fix_iteration']}] TOOL EXECUTED -> {tool_name}",
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
def should_continue(state: FixState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# ---------------- GRAPH ----------------
graph = StateGraph(FixState)
graph.add_node("llm", fix_llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", END])
graph.add_edge("tools", "llm")
fix_app = graph.compile()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Load existing RCA shared memory and message history
    shared_memory_path = "langgraph_rca_system/output/shared_memory.json"
    message_history_path = "langgraph_rca_system/output/message_history.json"

    with open(shared_memory_path, "r", encoding="utf-8") as f:
        shared_memory = json.load(f)

    with open(message_history_path, "r", encoding="utf-8") as f:
        message_history = json.load(f)

    # Initialize missing keys
    if "fix_iteration" not in shared_memory:
        shared_memory["fix_iteration"] = 0
    if "fix_result" not in shared_memory:
        shared_memory["fix_result"] = None

    result = fix_app.invoke({
        "messages": [],
        "shared_memory": shared_memory,
        "message_history": message_history
    })

    # Append final Fix result to message history
    result["message_history"].append({
        "event": "final_result",
        "iteration": result["shared_memory"]["fix_iteration"],
        "content": result["shared_memory"]["fix_result"]
    })

    log_console("FINAL FIX OUTPUT", result["shared_memory"]["fix_result"])

    # Write back to same RCA files
    with open(shared_memory_path, "w", encoding="utf-8") as f:
        json.dump(result["shared_memory"], f, indent=2)

    with open(message_history_path, "w", encoding="utf-8") as f:
        json.dump(result["message_history"], f, indent=2)

    log_console("FILES UPDATED", {
        "message_history.json": "Updated",
        "shared_memory.json": "Updated"
    })
