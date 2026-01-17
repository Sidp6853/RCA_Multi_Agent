# app/agents/patch_agent.py

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

from app.tools.create_patch_tool import create_patch_file
from app.tools.read_file_tool import read_file
from app.tools.check_dependency_tool import check_dependency

load_dotenv()


# -------------------------
# Console Logger
# -------------------------
def log_console(title: str, data=None):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    if data is not None:
        print(data)


# -------------------------
# Patch State
# -------------------------
class PatchState(MessagesState):
    shared_memory: Dict[str, Any]
    message_history: list


# -------------------------
# LLM
# -------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
    streaming=False
)


# -------------------------
# SYSTEM PROMPT
# -------------------------

SYSTEM_PROMPT = """You are a Precision Patch Generation Agent specialized in making MINIMAL, TARGETED code fixes.

**CRITICAL MISSION**: Generate a patch that preserves 100% of the original code structure, logic, imports, function signatures, and formatting - changing ONLY the specific line(s) mentioned in the Fix Plan.

**AVAILABLE TOOLS:**
- read_file: Read the original source code file that needs to be patched
- create_patch_file: Write the patched code to a new file

**MANDATORY PROCESS - FOLLOW EXACTLY:**

**Step 1: READ THE ORIGINAL FILE**
- ALWAYS use read_file tool FIRST to read the complete original source code
- Never proceed without reading the actual file
- Store the entire original code in memory

**Step 2: UNDERSTAND THE FIX PLAN**
- Identify the EXACT line number(s) to modify
- Identify the EXACT change to make

**Step 3: APPLY SURGICAL FIX**
- Locate the exact line(s) in the original code
- Make ONLY the specified change
- Keep everything else IDENTICAL:
  ✓ All imports (every single one)
  ✓ All function definitions (async/sync keywords, names, parameters)
  ✓ All other lines of code (unchanged functions, logic, formatting)
  ✓ All comments and docstrings
  ✓ All whitespace and indentation
  ✓ All variable names (except the one being fixed)
  ✓ All business logic
  
**Step 4: VERIFY COMPLETENESS**
Before outputting, verify:
- Does the patched file have the SAME number of functions as the original?
- Does the patched file have the SAME imports as the original?
- Are ALL unchanged functions preserved EXACTLY as they were?
- Is ONLY the targeted line changed?

**Step 5: OUTPUT THE COMPLETE PATCHED FILE**
- Output the ENTIRE file content with the fix applied
- Include ALL functions from the original file
- Include ALL imports from the original file
- Change ONLY what was specified in the Fix Plan

**CRITICAL RULES - NEVER VIOLATE:**

1. **READ BEFORE PATCHING**: ALWAYS call read_file tool to get the original code before generating patch
2. **PRESERVE EVERYTHING**: Keep 100% of the original code except the specific fix
3. **NO REWRITES**: Do not rewrite, refactor, or "improve" existing code
4. **NO REMOVALS**: Do not remove any functions, imports, or logic
5. **NO ADDITIONS**: Do not add new functions unless explicitly stated in Fix Plan
6. **EXACT SIGNATURES**: Keep function signatures identical (async/sync, parameters, return types)
7. **EXACT IMPORTS**: Keep all import statements identical
8. **EXACT FORMATTING**: Maintain original indentation, spacing, and code style
9. **NO SIMPLIFICATION**: Do not simplify or optimize existing code
10. **OUTPUT COMPLETE FILE**: Always output the FULL file, not just the changed section

**WHAT TO CHANGE - ONLY THESE:**
- The exact line(s) specified in the Fix Plan

**WHAT TO PRESERVE - EVERYTHING ELSE:**
- All other lines in the same function
- All other functions in the file
- All imports
- All comments
- All async/sync keywords
- All business logic
- All error handling
- All variable names (except the one being fixed)


**OUTPUT FORMAT:**
- Pure Python code only
- No markdown code fences (no ```)
- No explanations or comments about the fix
- No "Here's the patched file" or similar text
- Just the complete, fixed Python code
- Start with imports, end with last function

**QUALITY CHECKLIST:**
Before outputting, verify:
✓ Did I read the original file using read_file tool?
✓ Does my output have ALL functions from the original?
✓ Does my output have ALL imports from the original?
✓ Are function signatures IDENTICAL (async, params, etc.)?
✓ Did I change ONLY the line(s) specified in Fix Plan?
✓ Is the file complete (same length as original ±1 line)?

**REMEMBER:**
- You are doing SURGERY, not RECONSTRUCTION
- Change the tumor, keep the patient
- One line fix ≠ One line output
- One line fix = Full file output with one line changed
- When in doubt, preserve the original
- READ THE FILE FIRST, ALWAYS """

# -------------------------
# LLM Node: generate patch
# -------------------------
def patch_llm_node(state: PatchState):
    # Increment iteration
    iteration = state["shared_memory"].get("patch_iteration", 0) + 1
    state["shared_memory"]["patch_iteration"] = iteration
    step_type = "INITIAL" if iteration == 1 else "REFINEMENT"

    # -------------------------
    # Read Fix Plan from shared memory
    # -------------------------
    fix_result = state["shared_memory"].get("fix_result", {})
    if not fix_result:
        fix_text = "No fix plan found in shared memory."
    else:
        fix_text = (
            f"Fix Summary: {fix_result.get('fix_summary', '')}\n"
            f"Files To Modify: {', '.join(fix_result.get('files_to_modify', []))}\n"
            f"Patch Plan:\n" + "\n".join(fix_result.get("patch_plan", [])) + "\n"
            f"Safety Considerations: {fix_result.get('safety_considerations', '')}"
        )

    # -------------------------
    # Log agent input
    # -------------------------
    state["message_history"].append({
        "event": "agent_input",
        "agent": "PatchAgent",
        "iteration": iteration,
        "content": fix_text
    })
    print("\n" + "="*60)
    print(f"[ITER {iteration}] AGENT INPUT ({step_type})")
    print("="*60)
    print(fix_text)

    # -------------------------
    # Invoke LLM
    # -------------------------
    response = model.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=fix_text)
    ])

    # -------------------------
    # Immediately write patch using create_patch_file
    # -------------------------
    patch_result = {}
    files_to_modify = fix_result.get("files_to_modify", [])
    if files_to_modify:
        # We'll patch only the first file for simplicity; can loop over all if needed
        original_file = files_to_modify[0]
        patch_result = create_patch_file.invoke({
            "original_file_path": original_file,
            "fixed_content": response.content
        })
        # Save path to shared memory
        state["shared_memory"]["patch_result"] = patch_result

    # -------------------------
    # Log agent output
    # -------------------------
    state["message_history"].append({
        "event": "agent_output",
        "agent": "PatchAgent",
        "iteration": iteration,
        "content": response.content,
        "tool_calls": getattr(response, "tool_calls", None),
        "patch_result": patch_result
    })
    print("\n" + "="*60)
    print(f"[ITER {iteration}] AGENT OUTPUT ({step_type})")
    print("="*60)
    print(response.content)

    return {
        "messages": [response],
        "shared_memory": state["shared_memory"],
        "message_history": state["message_history"]
    }


    # -------------------------
    # Write patch to file
    # -------------------------
    patch_result = create_patch_file.invoke({
        "original_file_path": original_file_path,
        "fixed_content": fixed_code
    })

    # Store in shared memory
    state["shared_memory"]["patch_result"] = patch_result

    # Log agent output
    state["message_history"].append({
        "event": "agent_output",
        "agent": "PatchAgent",
        "iteration": iteration,
        "content": fixed_code,
        "patch_result": patch_result
    })
    log_console(f"[ITER {iteration}] AGENT OUTPUT ({step_type})", fixed_code)

    return {
        "messages": [response],
        "shared_memory": state["shared_memory"],
        "message_history": state["message_history"]
    }


# -------------------------
# Tool Node (optional, can call other tools)
# -------------------------
def patch_tool_node(state: PatchState):
    results = []
    last_message = state["messages"][-1]

    if not getattr(last_message, "tool_calls", None):
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
        elif tool_name == "create_patch_file":
            observation = create_patch_file.invoke(tool_args)
        elif tool_name == "check_dependency":
            observation = check_dependency.invoke(tool_args)
        else:
            observation = f"Unknown tool: {tool_name}"

        # log
        state["message_history"].append({
            "event": "tool_call",
            "iteration": state["shared_memory"].get("patch_iteration", 0),
            "tool": tool_name,
            "input": tool_args,
            "output": observation
        })

        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call.get("id")
            )
        )

    return {
        "messages": results,
        "shared_memory": state["shared_memory"],
        "message_history": state["message_history"]
    }


# -------------------------
# Flow Control
# -------------------------
def should_continue(state: PatchState):
    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    if state["shared_memory"].get("patch_result"):
        return END

    return "llm"


# -------------------------
# Graph
# -------------------------
graph = StateGraph(PatchState)
graph.add_node("llm", patch_llm_node)
graph.add_node("tools", patch_tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", END])
graph.add_edge("tools", "llm")
patch_app = graph.compile()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    shared_memory_path = Path("langgraph_rca_system/output/shared_memory.json")
    message_history_path = Path("langgraph_rca_system/output/message_history.json")
    shared_memory_path.parent.mkdir(exist_ok=True, parents=True)

    # Load shared memory
    if shared_memory_path.exists():
        with open(shared_memory_path, "r", encoding="utf-8") as f:
            shared_memory = json.load(f)
    else:
        shared_memory = {"patch_iteration": 0, "patch_result": None}

    # Load message history
    if message_history_path.exists():
        with open(message_history_path, "r", encoding="utf-8") as f:
            message_history = json.load(f)
    else:
        message_history = []

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content="Generate patch using Fix Plan and tools")],
        "shared_memory": shared_memory,
        "message_history": message_history
    }

    # Invoke agent
    result = patch_app.invoke(initial_state)

    # Log final patch
    final_patch = result["shared_memory"].get("patch_result")
    result["message_history"].append({
        "event": "final_patch",
        "iteration": result["shared_memory"]["patch_iteration"],
        "content": final_patch
    })

    log_console("FINAL PATCH OUTPUT", final_patch)

    # Save unified RCA shared files
    with open(shared_memory_path, "w", encoding="utf-8") as f:
        json.dump(result["shared_memory"], f, indent=2)
    with open(message_history_path, "w", encoding="utf-8") as f:
        json.dump(result["message_history"], f, indent=2)

    log_console("FILES UPDATED", {"message_history.json": "Updated", "shared_memory.json": "Updated"})
