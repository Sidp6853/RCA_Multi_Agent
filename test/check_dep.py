import os
from app.tools.check_dependency_tool import check_dependency

# -----------------------------
# SET YOUR CODEBASE ROOT
# -----------------------------

# IMPORTANT: must point to folder that contains your code files
os.environ["CODEBASE_ROOT"] = r"D:\Siddhi\projects\RCA-Agent\codebase"

# -----------------------------
# File you want to test
# -----------------------------

test_file = os.path.join(
    os.environ["CODEBASE_ROOT"],
    "app",
    "services",
    "user.py"
)    

# -----------------------------
# Run tool
# -----------------------------

result = check_dependency.invoke({"file_path": test_file})


# -----------------------------
# Pretty print result
# -----------------------------

print("\n===== CHECK DEPENDENCY RESULT =====\n")

if result["success"]:
    print("File:", result["file"])

    print("\nPython Dependencies:")
    for dep in result["python_dependencies"]:
        print(" -", dep)

    print("\nJS Dependencies:")
    for dep in result["js_dependencies"]:
        print(" -", dep)

else:
    print("ERROR:", result["error"])

print("\n===================================\n")
