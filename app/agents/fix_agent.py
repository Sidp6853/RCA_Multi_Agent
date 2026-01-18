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
import time

from app.tools.read_file_tool import read_file


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

# from langchain_groq import ChatGroq

# model = ChatGroq(
#     model="openai/gpt-oss-120b",
#     api_key=os.getenv("GROQ_API_KEY")
# )

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
tools = [read_file]
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

You MUST use the Affected File and Affected Line provided in RCA.
DO NOT introduce new files.
files_to_modify MUST contain ONLY the RCA affected file.


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
    time.sleep(30)
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

    fix_data = structured_fix.model_dump()

    # FORCE FIX FILE TO RCA AFFECTED FILE
    fix_data["files_to_modify"] = [rca_result["affected_file"]]

    state["shared_memory"]["fix_result"] = fix_data


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
            time.sleep(30)
            observation = read_file.invoke(tool_args)
        elif tool_name == "get_project_directory":
            time.sleep(30)
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



def should_continue(state: FixState):
    
    if state["shared_memory"].get("fix_iteration", 0) >= 3:
        return END

    last_message = state["messages"][-1]

    # If tools requested → go to tool node
    if last_message.tool_calls:
        return "tools"

    # Otherwise finish after fix is generated
    return END



# ---------------- GRAPH ----------------
graph = StateGraph(FixState)
graph.add_node("llm", fix_llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", END])
graph.add_edge("tools", "llm")
fix_app = graph.compile()