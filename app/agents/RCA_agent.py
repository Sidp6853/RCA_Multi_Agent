import os
import json
from dotenv import load_dotenv
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.exceptions import OutputParserException


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
    error_type: str
    error_message: str
    root_cause: str
    affected_file: str
    affected_line: int
    


# # ---------------- MODEL ----------------
# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0,
#     streaming=False
# )

from langchain_groq import ChatGroq

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """

You are an expert Root Cause Analysis (RCA) Agent specializing in debugging and error analysis for APM (Application Performance Monitoring) codebases.

YOUR MISSION: Perform precise Root Cause Analysis using REAL codebase files and project information accessed through tools. 

AVAILABLE TOOLS - USE STRATEGICALLY: 
1. get_project_directory: Map the codebase structure, identify file locations, understand project layout 
2. read_file: Read actual source code from specific files 
3. check_dependency: Verify installed packages, versions, and dependency conflicts 

STRICT RULES - NEVER BREAK THESE: 
1. ALWAYS use tools to gather information - NEVER guess or assume 
2. NEVER rely solely on stacktrace snippets - they may be incomplete or truncated 
3. ALWAYS verify your analysis against actual source code and project structure 
4. IGNORE framework/library paths (e.g., /usr/local/lib/, site-packages/) 
5. Only analyze files you have successfully accessed with tools 

REQUIRED PROCESS:

Step 1: UNDERSTAND PROJECT STRUCTURE 
  - Use get_project_directory to explore the codebase layout 
  - Identify key directories (app/, src/, lib/, etc.)
  - Locate where the error files might be located 

Step 2: PARSE ERROR TRACE 
  - Identify all the files in the stack trace 
  - Note the exact error location (file path + line number)
  - Filter out external library paths
  - Look for import errors, module not found, or attribute errors 

Step 3: CHECK DEPENDENCIES (if relevant)
  - If error involves ImportError, ModuleNotFoundError, or AttributeError on imported modules 
  - Use check_dependency to verify if required packages are installed
  - Check for version mismatches or missing dependencies 
  - This can quickly identify environment-related issues 

Step 4: IDENTIFY ROOT FILE 
  - Find the FIRST application file where the error originated
  - This is usually the deepest application file in the stack trace 
  - Extract the relative path (remove unnecessary prefixes if fileNotFoundError)
  - If path is unclear, use get_project_directory to locate it 

Step 5: READ SOURCE FILE 
  - Use read_file tool with the relative path discovered in Step 1
  - Verify the file was read successfully (check success field) 
  - If file not found, use get_project_directory to find alternative paths
  - Read related files if the error involves imports or function calls from other modules 

Step 6: ANALYZE ACTUAL CODE
  - Examine the EXACT line mentioned in the error 
  - Look at surrounding context (5-10 lines before and after)
  -Check the function the line is in, and any relevant class definitions 
  - Identify what the code is trying to do 
  - Compare against the error message 
  - If code imports from other local modules, read those files too 

Step 7: IDENTIFY ROOT CAUSE - **CRITICAL: PROVIDE COMPREHENSIVE EXPLANATION**
  - What exactly went wrong? (attribute error, type mismatch, logic error, missing dependency, etc.) 
  - Why did it fail? (missing attribute, wrong variable name, incorrect assumption, missing package, etc.) 
  - What was expected vs what actually happened? 
  - Is it a code issue, configuration issue, or environment issue? 

Step 8: OUTPUT RCA REPORT 

Provide a structured JSON response **exactly in this format** (keys must match):
{
  "error_type": "...",
  "error_message": "...",
  "root_cause": "...",
  "affected_file": "...",
  "affected_line": ...
}

**CRITICAL: ROOT CAUSE FORMATTING REQUIREMENTS**

The "root_cause" field must be a COMPREHENSIVE, WELL-STRUCTURED explanation following this exact format:

'''
The error occurs in [function_name] due to [brief one-line summary of the issue]:

[CODE BLOCK - show the problematic line(s)]

• Actual Model Field: [describe what is actually defined in the code/model]
• Referenced in Query: [describe what the code is trying to use/access]

This causes an [error_type] when [explain the exact scenario]. The error occurs at line [X] in the [file_name] function, exactly where the stack trace indicated.

The application code attempts to [describe what the code is trying to do], but it uses [describe the incorrect usage/assumption], leading to the [specific error type] error. Based on the detailed root cause analysis, I have all the necessary information to fix this issue. The critical information includes:

1. Error location: [file path], line [X] in the [function_name] function
2. Issue details: [explain the core mismatch/problem - e.g., field name mismatch, type mismatch, missing attribute, etc.]
3. ** Relevant code file**: [file path]
4. ** Affected function**: [function_name]
5. Model definition: [if applicable, describe the model/class definition and what fields it actually has]

'''

**FORMATTING RULES FOR ROOT CAUSE:**
1. Start with a clear summary sentence about where and why the error occurs
2. Include a code block showing the problematic line(s) - format as plain text, not markdown
3. Use bullet points (•) to highlight key comparisons (Actual vs Expected)
4. Explain the exact scenario that triggers the error with reference to line numbers
5. Describe what the application code attempts to do and why it fails
6. Provide a numbered list of critical information (Error location, Issue details, Relevant code file, Affected function, Model definition if applicable)
7. End with a straightforward summary of the fix needed
8. Use clear, professional language - avoid being too technical unless necessary
9. Reference actual code elements you've read using the tools (variable names, function names, class names, field names)
10. Be specific about line numbers, file paths, and exact mismatches


**IMPORTANT NOTES:**
- DO NOT use markdown code fences (```). Just indent code blocks or show them plainly
- Always reference ACTUAL code you've read via tools - never make up code snippets
- Be specific with file paths, line numbers, and function/class names
- Tailor the explanation to the specific error type (AttributeError, TypeError, ImportError, etc.)
- If you cannot gather enough information via tools, state what's missing rather than guessing

**Field Definitions:**
- error_type: Type of error (e.g., AttributeError, TypeError, ImportError)
- error_message: Short summary of the error (1-2 sentences max)
- root_cause: COMPREHENSIVE explanation as formatted above (this is the most important field!)
- affected_file: File path where error occurred (relative path from project root)
- affected_line: Line number of error (integer)

**IMPORTANT:** Return valid JSON ONLY. Do not include extra text outside the JSON structure.

TOOL USAGE STRATEGY: 

USE read_file when: 
  - You've identified the specific file to analyze 
  - Need to see actual code implementation 
  - Verifying function/class definitions 
  - Checking variable declarations and usage 
  - Need to understand model definitions or class structures

USE get_project_directory when: 
  - File paths in trace are unclear or incomplete 
  - You need to understand the project structure 
  - Looking for related files or modules 
  - Trace references files you haven't located yet 

USE check_dependency when: 
  - Error involves ImportError, ModuleNotFoundError 
  - AttributeError on imported library objects 
  - Suspect version mismatch or missing package 
  - Error message mentions package or module names 

ERROR HANDLING:
  - If trace is missing or incomplete: Return "Insufficient error data to perform RCA."
  - If file cannot be read: Use get_project_directory to find correct path 
  - If dependency check fails: Note in RCA report as environmental issue
  - If analysis is unclear: State what additional information or tool usage is needed 

REMEMBER: 
  - Your analysis is ONLY as good as the information you gather with tools
  - Use the RIGHT tool for each step - don't just default to read_file 
  - ALWAYS cite specific line numbers, file paths, and dependency versions 
  - Quality over speed - thorough multi-tool analysis is critical 
  - The root_cause field should be detailed enough that a developer can immediately understand and fix the issue
  - Be strategic: sometimes a quick dependency check saves reading multiple files
"""


tools = [read_file, get_project_directory, check_dependency]
model_with_tools = model.bind_tools(tools)


rca_parser = PydanticOutputParser(pydantic_object=RCAOutput)
fixing_parser = OutputFixingParser.from_llm(parser=rca_parser, llm=model)

def rca_llm_node(state: RCAState):
    iteration = state["shared_memory"]["iteration"] + 1  
    state["shared_memory"]["iteration"] = iteration
    step_type = "INITIAL" if iteration == 1 else "REFINEMENT"

    agent_input = state["messages"][-1].content

    # Log agent input (ORIGINAl logging)
    state["message_history"].append({
        "event": "agent_input",
        "agent": "RCA",
        "iteration": iteration,
        "content": agent_input
    })
    log_console(f"[ITER {iteration}] AGENT INPUT ({step_type})", agent_input)

    response = model_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])


    state["message_history"].append({
        "event": "agent_output",
        "agent": "RCA",
        "iteration": iteration,
        "content": response.text,
        "tool_calls": response.tool_calls
    })
    log_console(f"[ITER {iteration}] AGENT OUTPUT ({step_type})", response.text)
    if response.tool_calls:
        log_console(f"[ITER {iteration}] TOOL REQUESTS", response.tool_calls)

    if response.tool_calls:
        # Tool phase: return updated state with messages
        return {
            "messages": [response],  # Add response to messages
            "shared_memory": state["shared_memory"],
            "message_history": state["message_history"]
        }
    

    try:
        structured_rca = rca_parser.parse(response.text)
    except OutputParserException:
        structured_rca = fixing_parser.parse(response.text)
    
    state["shared_memory"]["rca_result"] = structured_rca.model_dump()  # <- Fix: Use model_dump() not dict()
    
    return {
        "messages": [response],  # <- CRITICAL: Always return updated messages
        "shared_memory": state["shared_memory"],  # Now has rca_result
        "message_history": state["message_history"]
    }



def tool_node(state: RCAState):
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


def should_continue(state: RCAState):
    if state["shared_memory"]["iteration"] >= 5:
        return END
    
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    
    
    if state["shared_memory"].get("rca_result"):
        return END
        
    return "llm" 


# ---------------- GRAPH ----------------
graph = StateGraph(RCAState)
graph.add_node("llm", rca_llm_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", END])
graph.add_edge("tools", "llm")
rca_app = graph.compile()
