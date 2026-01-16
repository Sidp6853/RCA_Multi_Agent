import os
from dotenv import load_dotenv
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

from app.memory.message_history import MessageHistoryLogger
from app.tools.read_file_tool import read_file
from app.tools.get_project_directory_tool import get_project_directory

load_dotenv()

# -------------------------------------------------
# MESSAGE LOGGER
# -------------------------------------------------
LOG_PATH = "logs/fix_agent_message_history.json"
history_logger = MessageHistoryLogger(LOG_PATH)

# -------------------------------------------------
# STATE
# -------------------------------------------------
class FixState(MessagesState):
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

Fix Summary:
- Short description

Files To Modify:
- file_path

Patch Plan:
1. Step-by-step fix instructions

Safety Considerations:
- Edge cases
- Regression risk
- Validation steps
"""

# -------------------------------------------------
# TOOL BINDING
# -------------------------------------------------
tools = [read_file, get_project_directory]
model_with_tools = model.bind_tools(tools)

# -------------------------------------------------
# LLM NODE
# -------------------------------------------------
def fix_llm_node(state: FixState):
    iteration = state["shared_memory"]["fix_iteration"] + 1
    state["shared_memory"]["fix_iteration"] = iteration

    rca_output = state["shared_memory"].get("final_result")
    if not rca_output:
        raise ValueError("RCA output missing in shared memory")

    fix_input = f"""
Root Cause Analysis Result:

{rca_output}

Generate Fix Plan:
"""

    # Invoke model
    response = model_with_tools.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=fix_input)
        ]
    )

    # Save internally
    state["shared_memory"]["fix_outputs"].append({
        "iteration": iteration,
        "content": response.content,
        "tool_calls": response.tool_calls
    })

    # -------- Log iteration --------
    history_logger.log_iteration(
        iteration=iteration,
        agent_name="fix_agent",
        input_data=fix_input,
        tool_calls=response.tool_calls,
        output_data=response.content
    )

    return {
        "messages": [response],
        "shared_memory": state["shared_memory"]
    }

# -------------------------------------------------
# TOOL EXECUTOR NODE
# -------------------------------------------------
def tool_node(state: FixState):
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

        # Save internally
        state["shared_memory"]["fix_tool_calls"].append({
            "iteration": state["shared_memory"]["fix_iteration"],
            "tool": tool_name,
            "input": tool_args,
            "output": observation
        })

        # Log tool execution
        history_logger.log_iteration(
            iteration=state["shared_memory"]["fix_iteration"],
            agent_name="fix_agent_tool",
            input_data=tool_args,
            tool_calls=[],
            output_data=observation
        )

        results.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"]
        ))

    return {
        "messages": results,
        "shared_memory": state["shared_memory"]
    }

# -------------------------------------------------
# ROUTER
# -------------------------------------------------
def should_continue(state: FixState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# -------------------------------------------------
# GRAPH
# -------------------------------------------------
graph = StateGraph(FixState)
graph.add_node("llm", fix_llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", END])
graph.add_edge("tools", "llm")

fix_app = graph.compile()

# -------------------------------------------------
# TEST RUNNER
# -------------------------------------------------
if __name__ == "__main__":
    # Simulate RCA output
    dummy_rca = """
AttributeError: User.emails does not exist.
Fix required: Change User.emails to User.email in user.py
"""

    # Initialize shared memory
    shared_memory = {
        "final_result": dummy_rca,
        "fix_iteration": 0,
        "fix_outputs": [],
        "fix_tool_calls": []
    }

    # Log initial RCA input
    history_logger.log_iteration(
        iteration=0,
        agent_name="user",
        input_data=dummy_rca,
        tool_calls=[],
        output_data=""
    )

    result = fix_app.invoke({
        "messages": [],
        "shared_memory": shared_memory
    })

    print("\n============= FIX PLAN OUTPUT =============\n")
    print(result["messages"][-1].content)
    print("\n==========================================\n")

    # Mark session complete
    history_logger.mark_complete()
