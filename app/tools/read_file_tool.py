import os
from typing import Dict, Any
from langchain.tools import tool


@tool("read_file")
def read_file(file_path: str) -> Dict[str, Any]:
    """
    Reads file content and returns structured response.
    
    Args:
        file_path: Path to file (relative or absolute)
        
    Returns:
        Dict with success status, content, and metadata
    """
    try:
        codebase_root = os.getenv("CODEBASE_ROOT")
        
        if not codebase_root:
            return {
                "success": False,
                "error": "CODEBASE_ROOT environment variable not set",
                "file_path": file_path
            }

        abs_path = os.path.join(codebase_root, file_path.lstrip("/"))

        abs_path = os.path.abspath(abs_path)

        # Checking if file exists or not 
        if not os.path.isfile(abs_path):
            return {
                "success": False,
                "error": f"File not found: {abs_path}",
                "file_path": file_path
            }

        #Reading the file content 
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
             
        
        return {
            "success": True,
            "file_path": file_path,
            "absolute_path": abs_path,
            "content": content,
            "lines": len(content.split('\n'))
            
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading file: {str(e)}",
            "file_path": file_path
        }