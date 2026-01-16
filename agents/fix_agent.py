# agents/fix_agent.py
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph_swarm import create_handoff_tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY")
)

fix_agent = create_agent(
    llm,
    tools=[
        create_handoff_tool(
            agent_name="PatchAgent",
            description="Send fix plan to Patch Generator"
        )
    ],
    system_prompt="""
You are Fix Suggestion Agent.

Responsibilities:
- Read RCA output from shared memory
- Generate clear fix plan
- Mention safety considerations
- Describe what patch must change

Rules:
- Do NOT write code
- Produce step-by-step fix plan
- After finishing, transfer to PatchAgent
""",
    name="FixAgent",
    
)
