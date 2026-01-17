import os
import json
from langchain.tools import tool
from pathlib import Path
from typing import Dict, Any


@tool
def get_project_directory(relative_path: str = ".") -> Dict[str, Any]:
    """
    List directory structure as a nested tree.
    Shows files and folders naturally stacked.
    
    Args:
        relative_path: Relative path from codebase root (default: ".")
        
    Returns:
        Dict with hierarchical tree structure
    """
    try:
        codebase_root = os.environ.get("CODEBASE_ROOT", "./codebase")
        root_path = Path(codebase_root) / relative_path
        
        if not root_path.exists():
            return {
                "success": False,
                "error": f"Directory not found: {relative_path}"
            }
        
        def build_tree(directory: Path) -> Dict:
            """Build tree structure using pathlib's iterdir()"""
            tree = {}
            
            try:
                # Use iterdir() to get immediate contents
                items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                
                for item in items:
                    # Skip hidden files
                    if item.name.startswith('.'):
                        continue
                    
                    if item.is_dir():
                        # Recursively build subtree
                        tree[f"{item.name}/"] = build_tree(item)
                    else:
                        # Just store filename
                        tree[item.name] = "file"
            
            except PermissionError:
                pass
            
            return tree
        
        return {
            "success": True,
            "structure": build_tree(root_path)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }