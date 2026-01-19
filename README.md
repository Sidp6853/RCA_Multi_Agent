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
├── streamlit_ui.py                         # Streamlit interactive UI
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



---

### Tool 2: `get_project_directory`

**Purpose:** Build recursive directory tree structure

**Used By:** RCA Agent

**Arguments:**
- `relative_path` (str): Path from CODEBASE_ROOT (default: ".")

**Max Depth:** 5 levels

---

### Tool 3: `check_dependency`

**Purpose:** Extract Python imports from source files

**Used By:** RCA Agent, Patch Agent

---

### Tool 4: `create_patch_file`

**Purpose:** Write fixed code to new file

**Used By:** Patch Agent

**Arguments:**
- `original_file_path` (str): Original file path
- `fixed_content` (str): Complete patched code

**Output Location:** `patches/fixed_<original_filename>`

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

## 🙏 Acknowledgments

- **LangGraph** - For state management and agent orchestration
- **Google Gemini** - For fast, reliable LLM inference
- **FastAPI** - For production-grade API framework

---

**Built for automated debugging of APM codebases** 🔧