import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

from app.tools.create_patch_tool import create_patch_file
from app.tools.read_file_tool import read_file
from app.tools.check_dependency_tool import check_dependency

load_dotenv()


def log_console(title: str, data=None):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    if data is not None:
        print(data)



class PatchState(MessagesState):
    shared_memory: Dict[str, Any]
    message_history: list



# base_model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     api_key=os.getenv("GOOGLE_API_KEY"),
#     temperature=0.2,
#     streaming=False
# )

from langchain_groq import ChatGroq

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)


tools = [read_file, check_dependency, create_patch_file]
model_with_tools = model.bind_tools(tools)


SYSTEM_PROMPT = """You are a Precision Patch Generation Agent specialized in making MINIMAL, TARGETED code fixes.


Before generating ANY code, you MUST:
1. **Call read_file tool** with the file path from "Files To Modify"
2. **Wait for the complete file content** to be returned
3. **ONLY THEN** generate the patched version

**If you do not call read_file first, your patch will be WRONG and INCOMPLETE.**


**CRITICAL MISSION**: 


Generate a patch that preserves 100% of the original code structure, logic, imports, function signatures, and formatting - changing ONLY the specific line(s) mentioned in the Fix Plan.

**MANDATORY PROCESS - FOLLOW EXACTLY:**

**Step 1: READ THE ORIGINAL FILE**
- ALWAYS use read_file tool FIRST to read the complete original source code
- Never proceed without reading the actual file
- Store the entire original code in memory

**Step 2: UNDERSTAND THE FIX PLAN**
- Identify the EXACT line number(s) to modify
- Identify the EXACT change to make
- Example: "Line 18: Change `User.emails` to `User.email`"

**Step 3: APPLY SURGICAL FIX**
- Locate the exact line(s) in the original code
- Make ONLY the specified change
- Keep EVERYTHING ELSE IDENTICAL:
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


CRITICAL RULES - NEVER VIOLATE:


1. **READ BEFORE PATCHING**: ALWAYS call read_file tool to get the original code
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

═══════════════════════════════════════════════════════════════════════════
📤 OUTPUT FORMAT:
═══════════════════════════════════════════════════════════════════════════

- Pure Python code only
- No markdown code fences (no ```)
- No explanations or comments about the fix
- No "Here's the patched file" or similar text
- Just the complete, fixed Python code
- Start with imports, end with last function

═══════════════════════════════════════════════════════════════════════════
🔍 QUALITY CHECKLIST:
═══════════════════════════════════════════════════════════════════════════

Before outputting, verify:
✓ Did I call read_file tool to read the original file?
✓ Does my output have ALL functions from the original?
✓ Does my output have ALL imports from the original?
✓ Are function signatures IDENTICAL (async, params, etc.)?
✓ Did I change ONLY the line(s) specified in Fix Plan?
✓ Is the file complete (same length as original ±1 line)?

═══════════════════════════════════════════════════════════════════════════
💡 REMEMBER:
═══════════════════════════════════════════════════════════════════════════

- You are doing SURGERY, not RECONSTRUCTION
- Change the tumor, keep the patient
- One line fix ≠ One line output
- One line fix = Full file output with one line changed
- When in doubt, preserve the original
- **READ THE FILE FIRST, ALWAYS**"""


def patch_llm_node(state: PatchState):
    iteration = state["shared_memory"].get("patch_iteration", 0) + 1
    state["shared_memory"]["patch_iteration"] = iteration
    step_type = "INITIAL" if iteration == 1 else "REFINEMENT"

    rca_result = state["shared_memory"].get("rca_result", {})
    fix_result = state["shared_memory"].get("fix_result", {})

    if not fix_result:
        fix_text = "No fix plan found in shared memory."
    else:

        files_to_modify = fix_result.get('files_to_modify', [])
        target_file = files_to_modify[0] if files_to_modify else 'N/A'

        fix_text = f"""
ROOT CAUSE ANALYSIS RESULTS:

Error Type: {rca_result.get('error_type', 'N/A')}
Error Message: {rca_result.get('error_message', 'N/A')}
Affected File: {rca_result.get('affected_file', 'N/A')}
Affected Line: {rca_result.get('affected_line', 'N/A')}

Root Cause Analysis:
{rca_result.get('root_cause', 'N/A')}

Fix Recommendation:
{rca_result.get('fix_recommendation', 'N/A')}

FIX PLAN:

Fix Summary: {fix_result.get('fix_summary', '')}

Files To Modify: {', '.join(files_to_modify)}

Step-by-Step Patch Plan:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(fix_result.get("patch_plan", [])))}

Safety Considerations:
{fix_result.get('safety_considerations', '')}

═══════════════════════════════════════════════════════════════════════════
YOUR TASK:
═══════════════════════════════════════════════════════════════════════════

1. **FIRST**: Call read_file tool with file_path="{target_file}"
2. **THEN**: Apply the fix at line {rca_result.get('affected_line', 'N/A')}
3. **FINALLY**: Output the COMPLETE file with ONLY that specific line changed

**DO NOT SKIP STEP 1 - You MUST read the file first!**
"""

    state["message_history"].append({
        "event": "agent_input",
        "agent": "PatchAgent",
        "iteration": iteration,
        "content": fix_text
    })

    log_console(f"[ITER {iteration}] AGENT INPUT ({step_type})", fix_text)

    try:

        # -------------------------------------------------
        # FORCE FILE READ ON FIRST ITERATION
        # -------------------------------------------------

        if iteration == 1 and not state["shared_memory"].get("patch_file_loaded"):
            log_console("📂 FORCING FILE READ", target_file)

            return {
                "messages": [
                    HumanMessage(content=f"Call read_file with file_path='{target_file}'")
                ],
                "shared_memory": state["shared_memory"],
                "message_history": state["message_history"]
            }

        response = model_with_tools.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=fix_text)
        ])

        if not hasattr(response, 'content') or response.content is None:
            error_msg = "LLM response blocked or missing content (likely safety filter)"
            log_console("⚠️ ERROR", error_msg)

            state["shared_memory"]["patch_result"] = {
                "success": False,
                "error": error_msg,
                "patch_file_path": None
            }

            state["message_history"].append({
                "event": "agent_output",
                "agent": "PatchAgent",
                "iteration": iteration,
                "content": error_msg,
                "tool_calls": None,
                "patch_result": state["shared_memory"]["patch_result"]
            })

            return {
                "messages": [AIMessage(content=error_msg)],
                "shared_memory": state["shared_memory"],
                "message_history": state["message_history"]
            }

        content = str(response.content) if response.content else ""
        tool_calls = getattr(response, "tool_calls", None)

        log_console(f"[ITER {iteration}] AGENT OUTPUT ({step_type})", content[:500] if content else "No content")

        if tool_calls:
            log_console(f"[ITER {iteration}] TOOL REQUESTS", tool_calls)

        if tool_calls:
            state["message_history"].append({
                "event": "agent_output",
                "agent": "PatchAgent",
                "iteration": iteration,
                "content": content,
                "tool_calls": tool_calls,
                "patch_result": None
            })

            return {
                "messages": [response],
                "shared_memory": state["shared_memory"],
                "message_history": state["message_history"]
            }

        patch_result = {}
        files_to_modify = fix_result.get("files_to_modify", [])

        if files_to_modify and content:
            original_file = files_to_modify[0]
            try:
                patch_result = create_patch_file.invoke({
                    "original_file_path": original_file,
                    "fixed_content": content
                })
                state["shared_memory"]["patch_result"] = patch_result
                log_console(f"[ITER {iteration}] ✅ PATCH CREATED", patch_result)
            except Exception as e:
                error_msg = f"Error creating patch file: {str(e)}"
                log_console("❌ PATCH CREATION ERROR", error_msg)
                patch_result = {"success": False, "error": error_msg}
                state["shared_memory"]["patch_result"] = patch_result
        else:
            patch_result = {
                "success": False,
                "error": "No files to modify or empty content"
            }
            state["shared_memory"]["patch_result"] = patch_result

        state["message_history"].append({
            "event": "agent_output",
            "agent": "PatchAgent",
            "iteration": iteration,
            "content": content,
            "tool_calls": None,
            "patch_result": patch_result
        })

        return {
            "messages": [response],
            "shared_memory": state["shared_memory"],
            "message_history": state["message_history"]
        }

    except Exception as e:

        error_msg = str(e)

        # -------------------------------
        # Handle API Quota / Rate Limit Gracefully
        # -------------------------------

        if "429" in error_msg or "quota" in error_msg.lower():

            state["message_history"].append({
                "event": "patch_llm_rate_limited",
                "timestamp": datetime.now().isoformat(),
                "iteration": iteration,
                "error": error_msg
            })

            log_console("⚠️ RATE LIMIT", "Patch LLM throttled — retrying next loop")

            return {
                "messages": [],
                "shared_memory": state["shared_memory"],
                "message_history": state["message_history"]
            }

        log_console("❌ PATCH FAILURE", error_msg)

        state["shared_memory"]["patch_result"] = {
            "success": False,
            "error": error_msg
        }

        state["message_history"].append({
            "event": "patch_generation_failed",
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "error": error_msg
        })

        return {
            "messages": [],
            "shared_memory": state["shared_memory"],
            "message_history": state["message_history"]
        }


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

        log_console(f"[TOOL EXECUTION] {tool_name}", tool_args)

        if tool_name == "read_file":
            observation = read_file.invoke(tool_args)
            state["shared_memory"]["patch_file_loaded"] = True

        elif tool_name == "check_dependency":
            observation = check_dependency.invoke(tool_args)

        else:
            observation = f"Unknown tool: {tool_name}"

        state["message_history"].append({
            "event": "tool_call",
            "iteration": state["shared_memory"].get("patch_iteration", 0),
            "tool": tool_name,
            "input": tool_args,
            "output": observation
        })

        log_console(f"[TOOL RESULT] {tool_name}", str(observation)[:500])

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


def should_continue(state: PatchState):

    patch_result = state["shared_memory"].get("patch_result")

    if patch_result and patch_result.get("success") is True:
        log_console("🏁 FLOW CONTROL", "Patch created successfully — ending workflow")
        return END

    if state["shared_memory"].get("patch_iteration", 0) >= 3:
        log_console("⚠️ FLOW CONTROL", "Max patch attempts reached — stopping")
        return END

    if state["messages"]:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            log_console("🔧 FLOW CONTROL", "Routing to tools")
            return "tools"

    log_console("🔄 FLOW CONTROL", "Retrying patch generation")
    return "llm"


graph = StateGraph(PatchState)
graph.add_node("llm", patch_llm_node)
graph.add_node("tools", patch_tool_node)
graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_continue, ["tools", "llm", END])
graph.add_edge("tools", "llm")
patch_app = graph.compile()