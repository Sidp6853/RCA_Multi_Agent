from app.tools.read_file_tool import read_file
from dotenv import load_dotenv
import os


load_dotenv()


codebase_root = os.getenv("CODEBASE_ROOT")

file_to_read = "app\services\email.py"

print("Codebase Root (ENV):", codebase_root)
print("Requested File:", file_to_read)
print("-" * 50)

try:
    
    content = read_file.run({
        "file_path": file_to_read
    })

    print("FILE CONTENT:\n")
    print(content)

except FileNotFoundError as e:
    print("ERROR:", e)

except Exception as e:
    print("UNEXPECTED ERROR:", e)
