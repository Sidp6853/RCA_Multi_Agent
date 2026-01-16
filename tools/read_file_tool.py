import os
from langchain.tools import tool


@tool("read_file")
def read_file(file_path: str) -> str:
    """
    Reads file content from project.
    """

    codebase_root = os.getenv("CODEBASE_ROOT")

    if not codebase_root:
        raise RuntimeError("CODEBASE_ROOT environment variable not set")

    # Normalize linux container path
    if file_path.startswith("/usr/srv/app"):
        relative_path = file_path.replace("/usr/srv/app", "").lstrip("/")
        abs_path = os.path.join(codebase_root, "app", relative_path)

    else:
        # fallback direct mapping
        abs_path = os.path.join(codebase_root, file_path.lstrip("/"))

    abs_path = os.path.abspath(abs_path)

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")

    with open(abs_path, "r", encoding="utf-8") as f:
        return f.read()
