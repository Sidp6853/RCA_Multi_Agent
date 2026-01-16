# agents/patch_agent.py
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.create_patch_tool import create_patch_file

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

patch_agent = create_agent(
    llm,
    tools=[create_patch_file],
    system_prompt="""
You are Patch Generation Agent.

Responsibilities:
- Read RCA output and Fix Plan from shared memory
- Generate the actual code fix
- Write the fix into a new file (use create_patch_file)
- Avoid hallucinations
- Apply minimal, safe changes
- Return patch metadata
""",
    name="PatchAgent",
    
)
