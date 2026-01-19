import os
from dotenv import load_dotenv

# Load .env (optional, if you use a .env file)
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    print("Current GOOGLE_API_KEY in environment:", api_key)
else:
    print("No GOOGLE_API_KEY is set in the environment.")
