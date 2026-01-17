import os
from dotenv import load_dotenv
from typing import Dict, Any


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
from app.tools.check_dependency_tool import check_dependency

load_dotenv()


class RCAState(MessagesState):
    shared_memory: Dict[str, Any]


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    streaming=False
)



SYSTEM_PROMPT = """
You are an expert Root Cause Analysis (RCA) Agent specializing in debugging and error analysis for APM (Application Performance Monitoring) codebases.

YOUR MISSION:
Perform precise Root Cause Analysis using REAL codebase files accessed through tools.

STRICT RULES - NEVER BREAK THESE:
1. ALWAYS read files using the read_file tool - NEVER guess file contents
2. NEVER rely solely on stacktrace snippets - they may be incomplete or truncated
3. ALWAYS verify your analysis against actual source code
4. Focus ONLY on application files (files under /usr/srv/app/ or similar app paths)
5. IGNORE framework/library paths (e.g., /usr/local/lib/, site-packages/)
6. Only analyze files you have successfully read with tools

REQUIRED PROCESS - FOLLOW STRICTLY:

Step 1: PARSE ERROR TRACE
   - Extract exception type, message, and line number
   - Identify all application files in the stack trace
   - Note the exact error location (file path + line number)
   - Filter out external library paths

Step 2: IDENTIFY ROOT FILE
   - Find the FIRST application file where the error originated
   - This is usually the deepest application file in the stack trace
   - Extract the relative path (remove /usr/srv/app/ or /usr/srv/ prefix)

Step 3: READ SOURCE FILE
   - Use read_file tool with the relative path
   - Verify the file was read successfully (check success field)
   - If file not found, try alternative paths (e.g., with/without 'app/' prefix)

Step 4: ANALYZE ACTUAL CODE
   - Examine the EXACT line mentioned in the error
   - Look at surrounding context (5-10 lines before and after)
   - Identify what the code is trying to do
   - Compare against the error message

Step 5: IDENTIFY ROOT CAUSE
   - What exactly went wrong? (attribute error, type mismatch, logic error, etc.)
   - Why did it fail? (missing attribute, wrong variable name, incorrect assumption, etc.)
   - What was expected vs what actually happened?

Step 6: OUTPUT RCA REPORT
   Provide a structured response with:
   - Error Summary: Brief description of what failed
   - Root Cause: Specific reason for failure with exact line reference
   - Evidence: Quote the problematic code line(s)
   - Fix Recommendation: Precise code change needed

ERROR HANDLING:
- If trace is missing or incomplete: Return "Insufficient error data to perform RCA."
- If file cannot be read: Try alternative paths, then report file access issue
- If analysis is unclear: State what additional information is needed

REMEMBER:
- Your analysis is ONLY as good as the files you actually read
- NEVER make assumptions about code you haven't seen
- ALWAYS cite specific line numbers and file paths
- Quality over speed - thorough analysis is critical
"""

tools = [read_file, get_project_directory, check_dependency]
model_with_tools = model.bind_tools(tools)



def rca_llm_node(state: RCAState):

    iteration = state["shared_memory"]["iteration"] + 1
    state["shared_memory"]["iteration"] = iteration

    response = model_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
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

        elif tool_name == "check_dependency":
            observation = check_dependency.invoke(tool_args)    

        else:
            observation = f"Unknown tool: {tool_name}"

   

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

def should_continue(state: RCAState):

    # Hard safety cap to avoid infinite loops
    if state["shared_memory"]["iteration"] >= 5:
        return END

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"

    return END


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



if __name__ == "__main__":

    os.environ["CODEBASE_ROOT"] = r"D:\Siddhi\projects\RCA-Agent\codebase"

    trace_path = r"D:\Siddhi\projects\RCA-Agent\trace_1.json"

    with open(trace_path, "r", encoding="utf-8") as f:
        trace_data = f.read()


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



    print("\n================ RCA OUTPUT ================\n")
    print(final_output)
    print("\n===========================================\n")