import json
from pathlib import Path
from datetime import datetime

class MessageHistoryLogger:

    def __init__(self, log_path: str = "logs/session_history.json"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)  # create logs folder if not exists
        self._initialize_file()

    def _initialize_file(self):
        if self.log_path.exists():
            return  # don't overwrite existing log

        base_structure = {
            "run_id": f"rca_pipeline_run_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "created_at": datetime.utcnow().isoformat(),
            "iterations": [],
            "final_status": "running"
        }

        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(base_structure, f, indent=4)

    def log_iteration(
        self,
        iteration: int,
        agent_name: str,
        input_data,
        tool_calls,
        output_data
    ):
        with open(self.log_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entry = {
            "iteration": iteration,
            "agent": agent_name,
            "input": input_data,
            "tool_calls": tool_calls,
            "output": output_data,
            "timestamp": datetime.utcnow().isoformat()
        }

        data["iterations"].append(entry)

        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def mark_complete(self):
        with open(self.log_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["final_status"] = "completed"

        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
