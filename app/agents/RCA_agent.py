import os
from dotenv import load_dotenv
from typing import Dict, Any

from app.memory.message_history import MessageHistoryLogger

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

from app.tools.read_file_tool import read_file
from app.tools.get_project_directory_tool import get_project_directory

load_dotenv()

# -------------------------------------------------
# MESSAGE LOGGER
# -------------------------------------------------

LOG_PATH = "logs/session_history.json"

history_logger = MessageHistoryLogger(LOG_PATH)


# -------------------------------------------------
# STATE
# -------------------------------------------------

class RCAState(MessagesState):
    shared_memory: Dict[str, Any]

# -------------------------------------------------
# MODEL
# -------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    streaming=False
)

# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------

SYSTEM_PROMPT = """
You are an expert Root Cause Analysis (RCA) Agent specializing in debugging and error analysis for APM (Application Performance Monitoring) codebases.

Your mission:
Perform a precise Root Cause Analysis using the REAL codebase files.

STRICT RULES:
- ALWAYS read files using tools
- NEVER guess
- NEVER rely only on stacktrace text
- Focus ONLY on application files (/usr/srv/app/)
- Ignore framework paths (/usr/local/lib/)

PROCESS:

1. Parse Error Trace
2. Identify root file
3. Explore project
4. Read exact source files
5. Compare expected vs actual
6. Output RCA

If trace is missing:
Return exactly:
Insufficient error data to perform RCA.
"""

# -------------------------------------------------
# TOOL BINDING
# -------------------------------------------------

tools = [read_file, get_project_directory]
model_with_tools = model.bind_tools(tools)

# -------------------------------------------------
# LLM NODE
# -------------------------------------------------

def rca_llm_node(state: RCAState):

    iteration = state["shared_memory"]["iteration"] + 1
    state["shared_memory"]["iteration"] = iteration

    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    )

   

    history_logger.log_iteration(
    iteration=iteration,
    agent_name="rca_agent",
    input_data=state["messages"][-1].content,
    tool_calls=state["shared_memory"]["tool_calls"],
    output_data=response.content
)


    # Save agent response internally
    state["shared_memory"]["agent_outputs"].append({
        "iteration": iteration,
        "content": response.content,
        "tool_calls": response.tool_calls
    })

    return {
        "messages": [response],
        "shared_memory": state["shared_memory"]
    }

# -------------------------------------------------
# TOOL EXECUTOR NODE
# -------------------------------------------------

def tool_node(state: RCAState):

    results = []
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return {"messages": [], "shared_memory": state["shared_memory"]}

    for tool_call in last_message.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "read_file":
            observation = read_file.invoke(tool_args)

        elif tool_name == "get_project_directory":
            observation = get_project_directory.invoke(tool_args)

        else:
            observation = f"Unknown tool: {tool_name}"

        # -------- Log Tool Execution --------
        history_logger.log(
            role="tool",
            content=str(observation),
            metadata={
                "tool": tool_name,
                "iteration": state["shared_memory"]["iteration"],
                "input": tool_args
            }
        )

        # Save internally
        state["shared_memory"]["tool_calls"].append({
            "iteration": state["shared_memory"]["iteration"],
            "tool": tool_name,
            "input": tool_args,
            "output": observation
        })

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            )
        )

    return {
        "messages": results,
        "shared_memory": state["shared_memory"]
    }

# -------------------------------------------------
# ROUTER WITH LOOP SAFETY
# -------------------------------------------------

def should_continue(state: RCAState):

    # Hard safety cap to avoid infinite loops
    if state["shared_memory"]["iteration"] >= 5:
        return END

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END

# -------------------------------------------------
# GRAPH
# -------------------------------------------------

graph = StateGraph(RCAState)

graph.add_node("llm", rca_llm_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "llm")

graph.add_conditional_edges(
    "llm",
    should_continue,
    ["tools", END]
)

graph.add_edge("tools", "llm")

rca_app = graph.compile()

# -------------------------------------------------
# RUNNER
# -------------------------------------------------

if __name__ == "__main__":

    os.environ["CODEBASE_ROOT"] = r"D:\Siddhi\projects\RCA-Agent\codebase"

    trace_path = r"D:\Siddhi\projects\RCA-Agent\trace_1.json"

    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = f.read()

    # -------- Log User Input --------
    history_logger.log(
        role="user",
        content=trace_data,
        metadata={"source": "trace_file"}
    )

    result = rca_app.invoke({
        "messages": [HumanMessage(content=trace_data)],
        "shared_memory": {
            "iteration": 0,
            "tool_calls": [],
            "agent_outputs": [],
            "final_result": None
        }
    })

    # Save final output
    final_output = result["messages"][-1].content
    result["shared_memory"]["final_result"] = final_output

    # -------- Log Final RCA --------
    history_logger.log(
        role="final_output",
        content=final_output
    )

    print("\n================ RCA OUTPUT ================\n")
    print(final_output)
    print("\n===========================================\n")
