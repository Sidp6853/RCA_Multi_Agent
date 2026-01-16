from tools.create_patch_tool import create_patch_file
from pathlib import Path

# Example buggy file path (just for naming, file need not exist)
original_file = "app/services/user.py"

# Example fixed content
fixed_code = """\
def add(a, b):
    return a + b

print(add(2, 3))
"""

try:
    # Run the LangChain tool
    result = create_patch_file.run({
        "original_file_path": original_file,
        "fixed_content": fixed_code
    })

    if result.get("success"):
        print("Patch File Created Successfully!\n")
        print(f"Original File: {result['original_file']}")
        print(f"Fixed Filename: {result['fixed_filename']}")
        print(f"Patch File Path: {result['absolute_path']}")
        print(f"File Size (bytes): {result['size_bytes']}")
        print(f"Number of Lines: {result['lines']}")
        print(f"Message: {result['message']}")
    else:
        print("Failed to create patch file:")
        print(result.get("error"))

except Exception as e:
    print("Unexpected error:", e)
