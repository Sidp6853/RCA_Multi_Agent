# main.py
import os
from langgraph_swarm import create_swarm
from langgraph.checkpoint.memory import InMemorySaver

# Import agents (already create_agent instances)
from agents.RCA import rca_agent       # RCA agent instance
from agents.fix_agent import fix_agent # Fix agent instance
from agents.patch_agent import patch_agent # Patch agent instance

# Import shared memory schema
from memory.shared_state import RCAState

# -------------------------
# Set CODEBASE_ROOT dynamically
os.environ["CODEBASE_ROOT"] = os.path.join(os.getcwd(), "codebase")

# Initialize shared memory
shared_state = RCAState()

# Initialize checkpointer
checkpointer = InMemorySaver()

# -------------------------
# Create the swarm with RCA → Fix → Patch
workflow = create_swarm(
    agents=[rca_agent, fix_agent, patch_agent],
    default_active_agent="RCAAgent",
    state_schema=RCAState
)

# Compile the workflow
app = workflow.compile(checkpointer=checkpointer)

# -------------------------
# Config for this pipeline run
config = {
    "configurable": {
        "thread_id": "rca_run_1"
    }
}

# -----------------------------
# File path for your trace
trace_file_path = r"D:\Siddhi\projects\RCA-Agent\trace_1.json"

# Read the actual trace content
# Read the trace content
with open(trace_file_path, "r", encoding="utf-8") as f:
    trace_content = f.read()

user_input = {
    "messages": [
        {
            "role": "user",
            "content": f"Analyze the following error trace and generate a fix patch:\n\n{trace_content}"
        }
    ]
}



# Run the pipeline
result = app.invoke(user_input, config)

# Extract AI message
ai_message = result['messages'][1].content

print("=== PIPELINE RESULT ===")
print(ai_message)

