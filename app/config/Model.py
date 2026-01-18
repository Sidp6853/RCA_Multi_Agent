import os
from langchain_google_genai import ChatGoogleGenerativeAI


class ModelConfig:

    @staticmethod
    def get_base_model() -> ChatGoogleGenerativeAI:
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Please export environment variable."
            )

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            temperature=0,
            streaming=False
        )


