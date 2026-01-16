import os
from typing import Dict, Any
from pathlib import Path
from langchain_core.tools import tool


@tool
def create_patch_file(
    original_file_path: str,
    fixed_content: str
) -> Dict[str, Any]:
    """
    Create a new file with fixed code. Names it as 'fixed_<original_name>'.
    
    Args:
        original_file_path: Path to original buggy file (e.g., "app/main.py")
        fixed_content: The complete fixed code content
        
    Returns:
        Dict with success status and patch file path
        
    Example:
        original_file_path: "app/models/user.py"
        creates: "patches/fixed_user.py"
    """
    try:
        # Create patches directory if it doesn't exist
        patches_dir = Path("patches")
        patches_dir.mkdir(exist_ok=True)
        
        # Get the original filename
        original_path = Path(original_file_path)
        original_name = original_path.name 
        
        # Create the fixed filename
        fixed_filename = f"fixed_{original_name}"
        patch_file_path = patches_dir / fixed_filename
        
        # Write the fixed content to the new file
        with open(patch_file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        # Get file stats
        file_size = patch_file_path.stat().st_size
        line_count = len(fixed_content.split('\n'))
        
        return {
            "success": True,
            "patch_file": str(patch_file_path),
            "original_file": original_file_path,
            "fixed_filename": fixed_filename,
            "size_bytes": file_size,
            "lines": line_count,
            "absolute_path": str(patch_file_path.absolute()),
            "message": f"Patch created successfully at {patch_file_path}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to create patch file: {str(e)}",
            "original_file": original_file_path
        }