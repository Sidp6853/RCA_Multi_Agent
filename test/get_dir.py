from tools.get_project_directory_tool import get_project_directory
import os

# Set the environment variable for testing
os.environ["CODEBASE_ROOT"] = os.path.abspath(os.path.join(os.getcwd(), "codebase"))

try:

    project_structure = get_project_directory.run({})
    print("Project Structure JSON:\n")
    print(project_structure)

except RuntimeError as e:
    print("ERROR:", e)
