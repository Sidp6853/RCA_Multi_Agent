import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


def get_gemini_model(
    model_name: str = "gemini-2.5-flash",
    streaming: bool = False
) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        streaming=streaming
    )


# Default model instance
MODEL = get_gemini_model()
