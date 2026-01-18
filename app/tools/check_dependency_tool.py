import re
from typing import Dict
from langchain.tools import tool
from app.tools.read_file_tool import read_file


@tool("check_dependency")
def check_dependency(file_path: str) -> Dict:
    """
    Analyze a source file and extract its import/dependency information.
    """
    #Currently this tool will only work for Python files.
    try:
        
        content = read_file.invoke({"file_path": file_path})

        python_imports = []

      

        python_import_pattern_1 = re.findall(
            r'^\s*import\s+([a-zA-Z0-9_\.]+)',
            content,
            re.MULTILINE
        )

        python_import_pattern_2 = re.findall(
            r'^\s*from\s+([a-zA-Z0-9_\.]+)\s+import',
            content,
            re.MULTILINE
        )

        python_imports.extend(python_import_pattern_1)
        python_imports.extend(python_import_pattern_2)

        

        python_imports = sorted(set(python_imports))
    

        return {
            "success": True,
            "file": file_path,
            "python_dependencies": python_imports
    
        }

    except Exception as e:
        return {
            "success": False,
            "file": file_path,
            "error": str(e)
        }
