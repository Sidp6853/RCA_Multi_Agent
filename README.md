# 🔍 Multi-Agent RCA System

**Automated Root Cause Analysis, Fix Suggestion & Patch Generation for APM Codebases**

A production-grade 3-agent AI system built with **LangGraph** and **Google Gemini 2.5 Flash** that performs end-to-end debugging: analyzing stack traces, generating fix plans, and creating patched code files.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Assignment Requirements](#assignment-requirements)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Agent Details](#agent-details)
- [Tool Documentation](#tool-documentation)
- [Submission Artifacts](#submission-artifacts)
- [Usage Examples](#usage-examples)
- [Design Decisions](#design-decisions)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This system orchestrates **3 specialized AI agents** to perform automated debugging of APM (Application Performance Monitoring) codebases:

1. **RCA Agent** - Analyzes stack traces and identifies root causes
2. **Fix Suggestion Agent** - Generates actionable fix plans with safety considerations
3. **Patch Generation Agent** - Creates patched code files with minimal, safe changes

### Key Capabilities

✅ **3-Agent Sequential Workflow** - RCA → Fix → Patch with shared memory state passing  
✅ **Strategic Tool Usage** - 4 custom tools distributed across agents  
✅ **Complete Message History** - Full audit trail of all agent interactions and tool calls  
✅ **Shared Memory** - Consistent JSON state object across all agents  
✅ **Production API** - FastAPI server with complete workflow orchestration  
✅ **Anti-Hallucination Design** - Tool-enforced verification and structured outputs  

---

## 📝 Assignment Requirements

### ✅ Required Agents (3 Agents Only)

| Agent | Responsibilities | Output Storage |
|-------|-----------------|----------------|
| **1️⃣ RCA Agent** | • Analyze stack traces and logs<br>• Identify root cause<br>• Identify affected file and code area<br>• Provide supporting evidence | Shared Memory |
| **2️⃣ Fix Suggestion Agent** | • Read RCA output<br>• Generate actionable fix plan<br>• Include safety considerations<br>• Describe patch requirements | Shared Memory |
| **3️⃣ Patch Generation Agent** | • Read RCA and Fix plan<br>• Use tools to interact with codebase<br>• Generate actual code fix<br>• Write to new file (`fixed_<original>.py`)<br>• Apply minimal, safe changes | Shared Memory |

### ✅ Technical Requirements Met

**Framework:** LangGraph (StateGraph-based orchestration)

**Tool Usage:**
- ✅ Multiple tool calls across workflow
- ✅ 4 different tool types implemented
- ✅ File operations coordinated across agents
- ✅ Tools used in all 3 agents

**Tools Built:**
1. `read_file` - File content reader
2. `get_project_directory` - Directory structure mapper
3. `check_dependency` - Python import parser
4. `create_patch_file` - Patch file writer

**State Management:**
- ✅ Shared Memory: JSON state object with `rca_result`, `fix_result`, `patch_result`
- ✅ Message History: Complete log of all agent interactions, tool calls, inputs/outputs, iterations

### ✅ Submission Artifacts

1. **✅ Complete Multi-Agent Codebase** - All agent definitions, tools, orchestration logic
2. **✅ Complete Message History (JSON)** - `output/<timestamp>/message_history.json`
3. **✅ Final Shared Memory JSON** - `output/shared_memory.json`
4. **✅ Generated Patch File** - `patches/fixed_<original>.py`
5. **✅ Run Instructions** - See [Quick Start](#quick-start) section below

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                   LangGraph Workflow Pipeline                    │
│                  (Sequential State Graph)                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
┌──────────┐    ┌──────────┐    ┌──────────┐
│   RCA    │───▶│   Fix    │───▶│  Patch   │
│  Agent   │    │  Agent   │    │  Agent   │
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     │  Tools:       │  Tools:       │  Tools:
     │  • read_file  │  • read_file  │  • read_file
     │  • get_dir    │               │  • check_dep
     │  • check_dep  │               │  • create_patch
     │               │               │
     ▼               ▼               ▼
┌─────────────────────────────────────────┐
│          Shared Memory (JSON)            │
│  {                                       │
│    rca_result: {...},                    │
│    fix_result: {...},                    │
│    patch_result: {...}                   │
│  }                                       │
└─────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────┐
│      Message History (Complete Log)      │
│  • All agent inputs/outputs              │
│  • All tool call inputs/outputs          │
│  • Iteration numbers                     │
│  • Retries and improvements              │
└─────────────────────────────────────────┘
```

### Agent Flow
```
Input: Error Trace JSON (stack traces + metadata)
    │
    ▼
┌───────────────────────────────────────────────────────┐
│ 1️⃣ RCA Agent                                          │
│    • Parses stack trace                               │
│    • Uses get_project_directory to map codebase       │
│    • Uses read_file to examine affected code          │
│    • Uses check_dependency for import errors          │
│    • Outputs: error_type, root_cause, affected_file,  │
│               affected_line                           │
│    • Stores in: shared_memory["rca_result"]          │
└─────────────────────┬─────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────┐
│ 2️⃣ Fix Suggestion Agent                              │
│    • Reads shared_memory["rca_result"]                │
│    • Optionally uses read_file for verification       │
│    • Outputs: fix_summary, files_to_modify,           │
│               patch_plan (step-by-step),              │
│               safety_considerations                   │
│    • Stores in: shared_memory["fix_result"]          │
└─────────────────────┬─────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────┐
│ 3️⃣ Patch Generation Agent                            │
│    • Reads shared_memory["rca_result"]                │
│    • Reads shared_memory["fix_result"]                │
│    • Uses read_file to fetch original code            │
│    • Generates complete patched file                  │
│    • Uses create_patch_file to write output           │
│    • Outputs: patches/fixed_<original>.py             │
│    • Stores in: shared_memory["patch_result"]        │
└─────────────────────┬─────────────────────────────────┘
                      │
                      ▼
                  Success!
    • Patch file: patches/fixed_<original>.py
    • Message history: output/<timestamp>/message_history.json
    • Shared memory: output/shared_memory.json
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- Google Gemini API Key ([Get one here](https://ai.google.dev/))

### Setup Steps

1. **Clone the repository**
```bash
   git clone https://github.com/Sidp6853/RCA_Multi_Agent.git
   cd RCA_Multi_Agent
```

2. **Create virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
   # Create .env file
   cat > .env << EOF
   GOOGLE_API_KEY=your_gemini_api_key_here
   CODEBASE_ROOT=/path/to/buggy/codebase
   EOF
```

---

## ⚡ Quick Start

### Run Instructions (End-to-End Workflow)

#### Option 1: API Server (Recommended)
```bash
# 1. Set environment variables
export GOOGLE_API_KEY="your-api-key"
export CODEBASE_ROOT="/path/to/codebase"

# 2. Start FastAPI server
python api.py

# 3. In another terminal, submit analysis request
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "trace_file_path": "trace_1.json",
    "codebase_root": "codebase"
  }'

# 4. Check outputs
ls patches/               # Patch file
ls output/                # Message history + shared memory
```

**Expected Output:**
```json
{
  "success": true,
  "message": "Workflow completed successfully",
  "results": {
    "rca": {...},
    "fix": {...},
    "patch": {
      "success": true,
      "patch_file": "patches/fixed_user.py"
    }
  }
}
```

---

#### Option 2: Streamlit UI
```bash
# 1. Start Streamlit interface
streamlit run ui.py

# 2. Open browser at http://localhost:8501

# 3. Fill form:
#    - Trace File Path: trace_1.json
#    - Codebase Root: codebase

# 4. Click "Start Analysis"

# 5. View results in tabs (RCA, Fix Plan, Patch)
```

---

#### Option 3: Programmatic Usage
```python
from app.workflow import pipeline, PipelineState
from langchain_core.messages import HumanMessage
import os
import json

# Set environment
os.environ["CODEBASE_ROOT"] = "/path/to/codebase"
os.environ["GOOGLE_API_KEY"] = "your-api-key"

# Load error trace
with open("trace_1.json") as f:
    trace_data = f.read()

# Run pipeline
initial_state = PipelineState()
initial_state.messages = [HumanMessage(content=trace_data)]

result = pipeline.invoke(
    initial_state,
    config={"configurable": {"thread_id": "run-1"}}
)

# Access results
print("RCA Result:", result.rca_result)
print("Fix Plan:", result.fix_result)
print("Patch File:", result.patch_result['patch_file'])

# Get complete message history
history = pipeline.get_state(config)
print("Total messages:", len(history.values["messages"]))
```

---

## 📁 Project Structure
```
RCA_Multi_Agent/
├── app/
│   ├── agents/
│   │   ├── rca_agent.py          # 1️⃣ RCA Agent (8-step forensic process)
│   │   ├── fix_agent.py          # 2️⃣ Fix Suggestion Agent
│   │   └── patch_agent.py        # 3️⃣ Patch Generation Agent
│   ├── tools/
│   │   ├── read_file_tool.py            # Tool: Read file content
│   │   ├── get_project_directory_tool.py # Tool: Map directory structure
│   │   ├── check_dependency_tool.py      # Tool: Parse Python imports
│   │   └── create_patch_tool.py          # Tool: Write patch file
│   ├── prompts/
│   │   ├── rca.py                # RCA system prompt
│   │   ├── fix.py                # Fix suggestion prompt
│   │   └── patch.py              # Patch generation prompt
│   ├── config/
│   │   └── Model.py              # Gemini 2.5 Flash config
│   └── workflow.py               # LangGraph orchestrator
├── patches/                      # 📁 Output: Generated patch files
│   └── fixed_<original>.py       # Example: fixed_user.py
├── output/                       # 📁 Output: Logs and state
│   ├── <timestamp>/
│   │   └── message_history.json  # ✅ Complete message history
│   └── shared_memory.json        # ✅ Final shared memory state
├── codebase/                     # 📁 Input: Buggy codebase
│   └── services/
│       └── user.py               # Example buggy file
├── api.py                        # FastAPI production server
├── ui.py                         # Streamlit interactive UI
├── trace_1.json                  # 📁 Input: Error trace JSON
├── requirements.txt              # Python dependencies
├── .env                          # Environment config
└── README.md                     # This file
```

---

## 🤖 Agent Details

### 1️⃣ RCA Agent

**Purpose:** Root Cause Analysis with forensic precision

**Tools Available:**
- `read_file` - Read source code files
- `get_project_directory` - Map codebase structure
- `check_dependency` - Verify imports and packages

**Output Schema (Stored in Shared Memory):**
```json
{
  "error_type": "AttributeError",
  "error_message": "type object 'User' has no attribute 'emails'",
  "root_cause": "Comprehensive explanation with code context...",
  "affected_file": "services/user.py",
  "affected_line": 18
}
```

**Process (8 Steps):**
1. Parse error trace
2. Identify root file from stack
3. Read source file with tool
4. Analyze actual code at error line
5. Understand project structure
6. Check dependencies if needed
7. Identify root cause with evidence
8. Output structured RCA report

**Max Iterations:** 5  
**Storage:** `shared_memory["rca_result"]`

---

### 2️⃣ Fix Suggestion Agent

**Purpose:** Generate actionable fix plans with safety considerations

**Tools Available:**
- `read_file` - Verify code context if needed

**Input:** Reads `shared_memory["rca_result"]`

**Output Schema (Stored in Shared Memory):**
```json
{
  "fix_summary": "Change User.emails to User.email on line 18",
  "files_to_modify": ["services/user.py"],
  "patch_plan": [
    "Step 1: Open services/user.py",
    "Step 2: Navigate to line 18",
    "Step 3: Change User.emails to User.email",
    "Step 4: Verify the change"
  ],
  "safety_considerations": "Ensure email field exists in User model"
}
```

**Max Iterations:** 3  
**Storage:** `shared_memory["fix_result"]`

---

### 3️⃣ Patch Generation Agent

**Purpose:** Generate patched code with minimal, safe changes

**Tools Available:**
- `read_file` - Fetch original file content
- `check_dependency` - Verify imports if needed
- `create_patch_file` - Write patched file to `patches/fixed_<original>.py`

**Input:** 
- Reads `shared_memory["rca_result"]`
- Reads `shared_memory["fix_result"]`

**Output Schema (Stored in Shared Memory):**
```json
{
  "success": true,
  "patch_file": "patches/fixed_user.py",
  "original_file": "services/user.py",
  "size_bytes": 1234,
  "lines": 45
}
```

**Process:**
1. Request original file via `read_file` tool
2. Generate complete patched file (preserving all original code)
3. Change ONLY the line(s) specified in fix plan
4. Write to `patches/fixed_<original>.py` via `create_patch_file` tool

**Max Iterations:** 5  
**Storage:** `shared_memory["patch_result"]`

---

## 🛠️ Tool Documentation

### Tool 1: `read_file`

**Purpose:** Read file content with CODEBASE_ROOT resolution

**Used By:** All 3 agents

**Arguments:**
- `file_path` (str): Relative or absolute file path

**Returns:**
```python
{
    "success": bool,
    "file_path": str,
    "absolute_path": str,
    "content": str,          # Full file content
    "lines": int             # Line count
}
```

---

### Tool 2: `get_project_directory`

**Purpose:** Build recursive directory tree structure

**Used By:** RCA Agent

**Arguments:**
- `relative_path` (str): Path from CODEBASE_ROOT (default: ".")

**Returns:**
```python
{
    "success": bool,
    "structure": {
        "folder/": {
            "file1.py": "file",
            "subfolder/": {...}
        }
    }
}
```

**Max Depth:** 5 levels

---

### Tool 3: `check_dependency`

**Purpose:** Extract Python imports from source files

**Used By:** RCA Agent, Patch Agent

**Arguments:**
- `file_path` (str): File to analyze

**Returns:**
```python
{
    "success": bool,
    "file": str,
    "python_dependencies": ["module1", "module2", ...]
}
```

---

### Tool 4: `create_patch_file`

**Purpose:** Write fixed code to new file

**Used By:** Patch Agent

**Arguments:**
- `original_file_path` (str): Original file path
- `fixed_content` (str): Complete patched code

**Returns:**
```python
{
    "success": bool,
    "patch_file": "patches/fixed_<original>.py",
    "original_file": str,
    "size_bytes": int,
    "lines": int
}
```

**Output Location:** `patches/fixed_<original_filename>`

---

## 📦 Submission Artifacts

### 1. ✅ Complete Multi-Agent Codebase

**Location:** Entire repository

**Includes:**
- Agent definitions (`app/agents/`)
- Tool implementations (`app/tools/`)
- Orchestration logic (`app/workflow.py`)
- Shared memory & message logging (built into LangGraph)
- Supporting scripts (`api.py`, `ui.py`)

**How to Run:** See [Quick Start](#quick-start)

---

### 2. ✅ Complete Message History (JSON)

**Location:** `output/<timestamp>/message_history.json`

**Generated By:** API server automatically after each workflow run

**Schema:**
```json
{
  "thread_id": "20250119_120000_1234",
  "timestamp": "2025-01-19T12:00:00",
  "input": {
    "trace_file": "trace_1.json",
    "codebase_root": "codebase"
  },
  "complete_message_history": [
    {
      "role": "user",
      "content": "<error trace>",
      "tool_calls": [],
      "name": null
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [{"name": "read_file", "args": {...}}],
      "name": null
    },
    {
      "role": "tool",
      "content": "Successfully read file...",
      "name": "read_file"
    }
  ],
  "rca_result": {...},
  "fix_result": {...},
  "patch_result": {...},
  "stats": {
    "total_messages": 25,
    "tool_calls": 8,
    "success": true
  }
}
```

**Contains:**
- ✅ All agent inputs/outputs
- ✅ All tool call inputs/outputs
- ✅ Iteration numbers
- ✅ Retries and improvements

---

### 3. ✅ Final Shared Memory JSON

**Location:** `output/shared_memory.json`

**Generated By:** API server after workflow completion

**Schema:**
```json
{
  "rca_result": {
    "error_type": "AttributeError",
    "error_message": "type object 'User' has no attribute 'emails'",
    "root_cause": "...",
    "affected_file": "services/user.py",
    "affected_line": 18
  },
  "fix_result": {
    "fix_summary": "Change User.emails to User.email on line 18",
    "files_to_modify": ["services/user.py"],
    "patch_plan": [...],
    "safety_considerations": "..."
  },
  "patch_result": {
    "success": true,
    "patch_file": "patches/fixed_user.py",
    "original_file": "services/user.py"
  }
}
```

---

### 4. ✅ Generated Patch File

**Location:** `patches/fixed_<original>.py`

**Example:** `patches/fixed_user.py`

**Format:** Complete Python file with minimal changes

**Before (Line 18):**
```python
user_exist = session.query(User).filter(User.emails == data.email).first()
```

**After (Line 18):**
```python
user_exist = session.query(User).filter(User.email == data.email).first()
```

**Characteristics:**
- ✅ Complete file (all imports, functions preserved)
- ✅ Only targeted line changed
- ✅ No hallucinations or unnecessary modifications
- ✅ Minimal, safe changes as required

---

### 5. ✅ Run Instructions

See [Quick Start](#quick-start) section for:
- API Server setup (Option 1)
- Streamlit UI usage (Option 2)
- Programmatic execution (Option 3)

---

## 📚 Usage Examples

### Example: Fixing AttributeError

**Input File:** `trace_1.json`
```json
{
  "event_attributes": {
    "exception.message": "type object 'User' has no attribute 'emails'",
    "exception.type": "AttributeError",
    "exception.stacktrace": "...File '/usr/srv/app/services/user.py', line 18..."
  }
}
```

**Step 1: Run Workflow**
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "trace_file_path": "trace_1.json",
    "codebase_root": "codebase"
  }'
```

**Step 2: Agent 1 (RCA) Output**
```json
{
  "error_type": "AttributeError",
  "affected_file": "services/user.py",
  "affected_line": 18,
  "root_cause": "Line 18 uses User.emails but model defines User.email (singular)"
}
```

**Step 3: Agent 2 (Fix) Output**
```json
{
  "fix_summary": "Change User.emails to User.email on line 18",
  "files_to_modify": ["services/user.py"],
  "patch_plan": [
    "Navigate to services/user.py line 18",
    "Change User.emails to User.email in query filter"
  ]
}
```

**Step 4: Agent 3 (Patch) Output**
- **File Created:** `patches/fixed_user.py`
- **Change:** Line 18 modified
- **All other code:** Preserved exactly

**Step 5: Verify Artifacts**
```bash
ls patches/fixed_user.py               # ✅ Patch file
cat output/*/message_history.json      # ✅ Complete history
cat output/shared_memory.json          # ✅ Shared state
```

---

## 🎨 Design Decisions

### Why LangGraph?
- **State Management:** Built-in shared memory across agents
- **Checkpointing:** Automatic message history retention
- **Tool Coordination:** Native tool calling with conditional routing

### Why 3 Sequential Agents?
- **Clear Separation:** RCA → Fix → Patch follows logical debugging flow
- **State Dependencies:** Each agent builds on previous results
- **Auditability:** Easy to trace which agent made which decision

### Tool Distribution Strategy

| Tool | RCA Agent | Fix Agent | Patch Agent | Rationale |
|------|-----------|-----------|-------------|-----------|
| `read_file` | ✅ | ✅ | ✅ | All need file access |
| `get_project_directory` | ✅ | ❌ | ❌ | Only RCA needs codebase mapping |
| `check_dependency` | ✅ | ❌ | ✅ | RCA for analysis, Patch for verification |
| `create_patch_file` | ❌ | ❌ | ✅ | Only Patch writes output |

### Anti-Hallucination Measures

1. **Enforced Tool Usage:** Agents MUST call tools before generating outputs
2. **Structured Outputs:** Pydantic schemas prevent malformed responses
3. **Hardcoded Constraints:** Fix agent forces `files_to_modify = [rca_result["affected_file"]]`
4. **Two-Phase Patch Generation:** Read file → Generate patch (prevents inventing code)
5. **Comprehensive Prompts:** Detailed instructions with formatting rules

---

## 🐛 Troubleshooting

### Issue: "CODEBASE_ROOT environment variable not set"

**Solution:**
```bash
export CODEBASE_ROOT=/path/to/codebase
# or add to .env file
echo "CODEBASE_ROOT=/path/to/codebase" >> .env
```

---

### Issue: "File not found" from read_file tool

**Solution:**
- Verify `CODEBASE_ROOT` is correct
- Use relative paths (e.g., `services/user.py` not `/usr/srv/app/services/user.py`)
- Check file exists: `ls $CODEBASE_ROOT/services/user.py`

---

### Issue: Empty patch_result

**Debug:**
```bash
# Check logs
cat api.log | grep "PATCH"

# Check message history
cat output/*/message_history.json | jq '.patch_result'
```

**Common Causes:**
- Tool call failed (original file not found)
- Structured output parsing error
- Exception during code generation

---

### Issue: Gemini API quota errors (429)

**Solution:**
- Use API key with higher quota
- Increase `time.sleep()` delays in agent code
- Implement exponential backoff

---

## 📄 License

This project was created as part of an interview assignment.

---

## 🙏 Acknowledgments

- **LangGraph** - For state management and agent orchestration
- **Google Gemini** - For fast, reliable LLM inference
- **FastAPI** - For production-grade API framework

---

**Built for automated debugging of APM codebases** 🔧