import re
from typing import Dict
from langchain.tools import tool
from app.tools.read_file_tool import read_file


@tool("check_dependency")
def check_dependency(file_path: str) -> Dict:
    """
    Analyze a source file and extract its import/dependency information.
    """

    try:
        # IMPORTANT: invoke tool properly
        content = read_file.invoke({"file_path": file_path})

        python_imports = []
        js_imports = []

        # -------------------------
        # Python import patterns
        # -------------------------

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

        # -------------------------
        # JS import patterns
        # -------------------------

        js_import_pattern_1 = re.findall(
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            content
        )

        js_import_pattern_2 = re.findall(
            r'require\([\'"]([^\'"]+)[\'"]\)',
            content
        )

        js_imports.extend(js_import_pattern_1)
        js_imports.extend(js_import_pattern_2)

        # -------------------------
        # Cleanup duplicates
        # -------------------------

        python_imports = sorted(set(python_imports))
        js_imports = sorted(set(js_imports))

        return {
            "success": True,
            "file": file_path,
            "python_dependencies": python_imports,
            "js_dependencies": js_imports
        }

    except Exception as e:
        return {
            "success": False,
            "file": file_path,
            "error": str(e)
        }
