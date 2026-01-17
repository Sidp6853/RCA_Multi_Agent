# app/main.py
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

from app.agents.RCA_agent import graph as rca_graph
from app.agents.fix_agent import graph as fix_graph
from app.agents.patch_agent import PatchAgent
from app.memory.message_history import MessageHistoryLogger
from app.tools.read_file_tool import read_file

load_dotenv()

class PipelineState(MessagesState):
    shared_memory: dict


patch_agent = PatchAgent()

# Message logger for pipeline
pipeline_logger = MessageHistoryLogger("logs/pipeline_message_history.json")


def rca_node(state: PipelineState):
    # Run RCA agent
    rca_result = rca_graph.invoke({
        "messages": state["messages"],
        "shared_memory": state["shared_memory"]
    })

    # Save RCA output
    state["shared_memory"]["rca_output"] = rca_result["shared_memory"].get("final_result", "")

    # Log iteration
    pipeline_logger.log_iteration(
        iteration=state["shared_memory"].get("iteration", 0) + 1,
        agent_name="RCA Agent",
        input_data=state["shared_memory"].get("user_input"),
        output_data=state["shared_memory"]["rca_output"]
    )

    return {
        "messages": rca_result["messages"],
        "shared_memory": state["shared_memory"]
    }

def fix_node(state: PipelineState):
    # Run Fix agent
    fix_result = fix_graph.invoke({
        "messages": state["messages"],
        "shared_memory": state["shared_memory"]
    })

    # Save fix plan
    state["shared_memory"]["fix_plan"] = fix_result["shared_memory"]["fix_outputs"][-1]["content"]

    # Log iteration
    pipeline_logger.log_iteration(
        iteration=state["shared_memory"].get("iteration", 0) + 1,
        agent_name="Fix Agent",
        input_data=state["shared_memory"].get("rca_output"),
        output_data=state["shared_memory"]["fix_plan"]
    )

    return {
        "messages": fix_result["messages"],
        "shared_memory": state["shared_memory"]
    }

def patch_node(state: PipelineState):
    # Use read_file tool to get original file content
    original_file = state["shared_memory"].get("file_path")
    file_content = read_file.invoke({"file_path": original_file})

    # Generate patch
    patch_result = patch_agent.generate_patch(original_file, state["shared_memory"]["fix_plan"])

    # Save patch result
    state["shared_memory"]["patch_result"] = patch_result

    # Log iteration
    pipeline_logger.log_iteration(
        iteration=state["shared_memory"].get("iteration", 0) + 1,
        agent_name="Patch Agent",
        input_data={"file_path": original_file, "fix_plan": state["shared_memory"]["fix_plan"]},
        output_data=patch_result
    )

    return {
        "messages": [],
        "shared_memory": state["shared_memory"]
    }

graph = StateGraph(PipelineState)
graph.add_node("rca", rca_node)
graph.add_node("fix", fix_node)
graph.add_node("patch", patch_node)

graph.add_edge(START, "rca")
graph.add_edge("rca", "fix")
graph.add_edge("fix", "patch")
graph.add_edge("patch", END)

pipeline_app = graph.compile()

if __name__ == "__main__":
    user_input = "AttributeError: User.emails does not exist in user.py"

    shared_memory = {
        "user_input": user_input,
        "file_path": "codebase/user.py",
        "iteration": 0
    }

    result = pipeline_app.invoke({
        "messages": [],
        "shared_memory": shared_memory
    })

    print("\n=========== PATCH RESULT ===========\n")
    print(result["shared_memory"].get("patch_result"))
    print("\n===================================\n")

    pipeline_logger.mark_complete()
