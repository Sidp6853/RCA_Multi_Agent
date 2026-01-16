import json
import os
from dotenv import load_dotenv


from tools.read_file_tool import read_file
from tools.get_project_directory_tool import get_project_directory


load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

rca_agent = create_agent(
    llm,
    tools=[
         read_file, get_project_directory
    ],
    system_prompt="""
You are an RCA Agent.
Responsibilities:
- Analyze error trace
- Inspect project structure
- Identify root cause, affected file, and evidence

Approach:
1. use the `read_file` tool to read the content of the trace file.
IMPORTANT:
If the error trace is missing, empty, or unclear,
DO NOT guess.
Respond with:
"Insufficient error data to perform RCA."


""",
    name="RCAAgent",
    
)
