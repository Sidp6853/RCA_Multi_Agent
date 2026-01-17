import os
from dotenv import load_dotenv
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState


from app.tools.read_file_tool import read_file
from app.tools.get_project_directory_tool import get_project_directory

import json


load_dotenv()


class FixState(MessagesState):
    shared_memory: Dict[str, Any]


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    streaming=False
)


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

Generate the fix plan **only once**. Do not repeat or rephrase steps multiple times.

"""


tools = [read_file, get_project_directory]
model_with_tools = model.bind_tools(tools) 


def fix_llm_node(state: FixState):
    iteration = state["shared_memory"]["fix_iteration"] + 1
    state["shared_memory"]["fix_iteration"] = iteration

    rca_output = state["shared_memory"].get("rca_result")

    if not rca_output:
        raise ValueError("RCA output missing in shared memory")

    fix_input = f"""
            Root Cause Analysis:

            Error Type: {rca_output['error_type']}
            Error Message: {rca_output['error_message']}
            Root Cause: {rca_output['root_cause']}
            Affected File: {rca_output['affected_file']}
            Affected Line: {rca_output['affected_line']}

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

    return {
        "messages": [response],
        "shared_memory": state["shared_memory"]
    }


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


        results.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"]
        ))

    return {
        "messages": results,
        "shared_memory": state["shared_memory"]
    }


def should_continue(state: FixState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


graph = StateGraph(FixState)
graph.add_node("llm", fix_llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", END])
graph.add_edge("tools", "llm")

fix_app = graph.compile()


if __name__ == "__main__":
    
    with open("langgraph_rca_system/output/shared_memory.json", "r", encoding="utf-8") as f:
        rca_shared = json.load(f)

    rca_result = rca_shared["rca_result"] 


    shared_memory = {
    "rca_result": rca_result,
    "fix_iteration": 0,
    "fix_outputs": [],
    "fix_tool_calls": []
}




    result = fix_app.invoke({
        "messages": [],
        "shared_memory": shared_memory
    })

    print("\n============= FIX PLAN OUTPUT =============\n")
    print(result["messages"][-1].content)
    print("\n==========================================\n")

    
