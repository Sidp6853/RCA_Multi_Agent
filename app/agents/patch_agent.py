# app/agents/patch_agent.py

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.tools.create_patch_tool import create_patch_file

load_dotenv()


# -------------------------
# Message History Logger
# -------------------------
class MessageHistoryLogger:
    """Logs agent iterations to a JSON file for continuous session tracking."""

    def __init__(self, file_path: str = "logs/patch_agent_message_history.json"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(exist_ok=True, parents=True)

        if not self.file_path.exists():
            self._initialize_file()

    def _initialize_file(self):
        base_structure = {
            "run_id": f"patch_pipeline_run_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "created_at": datetime.utcnow().isoformat(),
            "iterations": [],
            "final_status": "running"
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(base_structure, f, indent=4)

    def log_iteration(self, iteration: int, agent_name: str, input_data, output_data):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entry = {
            "iteration": iteration,
            "agent": agent_name,
            "input": input_data,
            "output": output_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        data["iterations"].append(entry)

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def mark_complete(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["final_status"] = "completed"

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


# -------------------------
# Patch Agent
# -------------------------
class PatchAgent:
    def __init__(self, temperature=0.2):
        # Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            streaming=False
        )

        # Message history
        self.history_logger = MessageHistoryLogger()

        self.iteration = 0
        self.patches_dir = Path("patches")
        self.patches_dir.mkdir(exist_ok=True)

    def generate_patch(self, original_file_name: str, fix_plan: str) -> Dict[str, Any]:
        self.iteration += 1
        agent_name = "PatchAgent"

        # -------------------------
        # LLM Prompt
        # -------------------------
        system_prompt = (
            "You are a Patch Generation Agent.\n"
            "You receive a fix plan and must generate the fixed Python code.\n"
            "Output ONLY the fixed Python code. Do NOT add explanations, markdown, or comments."
        )

        # Call Gemini LLM
        llm_response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"File: {original_file_name}\nFix Plan:\n{fix_plan}")
        ])

        fixed_code = llm_response.content.strip()

        # -------------------------
        # Save patch safely
        # -------------------------
        patch_result = create_patch_file.invoke({
            "original_file_path": str(self.patches_dir / original_file_name),
            "fixed_content": fixed_code
        })

        # -------------------------
        # Log iteration
        # -------------------------
        self.history_logger.log_iteration(
            iteration=self.iteration,
            agent_name=agent_name,
            input_data={"file_name": original_file_name, "fix_plan": fix_plan},
            output_data=patch_result
        )

        return patch_result


# -------------------------
# Run Example
# -------------------------
if __name__ == "__main__":
    patch_agent = PatchAgent()

    # Example inputs
    original_file = "example_user.py"
    fix_plan = (
        "Class User has attribute 'emails' which should be renamed to 'email'.\n"
        "Ensure the constructor reflects this change."
    )

    result = patch_agent.generate_patch(original_file, fix_plan)
    print("\n========= PATCH RESULT =========")
    print(result)
    print("===============================")

    patch_agent.history_logger.mark_complete()
