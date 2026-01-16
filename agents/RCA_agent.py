# import langchain
# langchain.debug = True

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langgraph.graph import StateGraph, START, END, MessagesState

from tools.read_file_tool import read_file
from tools.get_project_directory_tool import get_project_directory

load_dotenv()



model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    streaming=True
)


system_prompt = """ You are an expert Root Cause Analysis (RCA) Agent specializing in debugging and error analysis for APM (Application Performance Monitoring) codebases.

**Your Mission:**
Perform a comprehensive root cause analysis of the provided error trace and codebase to identify the exact source and nature of the bug.
You MUST use the available tools to examine the actual codebase. Do NOT make assumptions or provide analysis without reading the actual files.

**Available Tools:**
- read_file: Read the contents of any file in the codebase
- get_project_directory: Explore the project structure to understand the codebase layout

**Your Analysis Process (FOLLOW IN ORDER):**

1. **Parse the Error Trace**: Extract the file path and line number where the error occurred
   - Look for the PRIMARY error location (not framework/library code)
   - In this case, focus on files in `/usr/srv/app/` NOT in `/usr/local/lib/`

2. **Read the Problematic File**:
   - Use `read_file` to read the EXACT file mentioned in the error trace
   - Example: If error is in `/usr/srv/app/services/user.py`, read that file

3. **Explore the Codebase Structure**: 
   - Use `get_project_directory` to understand the project layout
   - Identify which files are relevant to the error   

4. **Read Related Files**:
   - Use `read_file` to examine related files (models, routes, etc.)
   - For this error type, you MUST read the User model definition
   - Read any other files that provide context

5. **Analyze and Provide RCA**:
   - Only AFTER reading the files, provide your complete analysis


**Required Report Structure:**

"""



tools = [read_file, get_project_directory]

model_with_tools = model.bind_tools(tools)


def rca_llm_node(state: MessagesState):
    """
    Calls the LLM.
    Always injects system prompt to preserve behavior across loops.
    """

    response = model_with_tools.invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )

    return {"messages": [response]}




def tool_node(state: MessagesState):
    """
    Executes tool calls and injects ToolMessage results back to graph.
    """

    results = []
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return {"messages": []}

    for tool_call in last_message.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "read_file":
            observation = read_file.invoke(tool_args)

        elif tool_name == "get_project_directory":
            observation = get_project_directory.invoke(tool_args)

        else:
            observation = f"Unknown tool: {tool_name}"

        results.append(
            ToolMessage(
                content=observation,
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": results}




def should_continue(state: MessagesState):
    """
    Decide whether to continue looping or terminate.
    """

    last_message = state["messages"][-1]

    # If LLM requested tools → go to tool node
    if last_message.tool_calls:
        return "tools"

    # Otherwise finish
    return END




graph = StateGraph(MessagesState)

graph.add_node("llm", rca_llm_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "llm")

graph.add_conditional_edges(
    "llm",
    should_continue,
    ["tools", END]
)

graph.add_edge("tools", "llm")

app = graph.compile()




trace_json_path = r"D:\Siddhi\projects\RCA-Agent\trace_1.json"

with open(trace_json_path, "r", encoding="utf-8") as f:
    trace_data = f.read()



result = app.invoke(
    {
        "messages": [
            HumanMessage(content=trace_data)
        ]
    }
)



for msg in result["messages"]:
    print("\n---------------- MESSAGE ----------------")
    print(msg)
