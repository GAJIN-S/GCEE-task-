import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.6-flash"
SYSTEM_PROMPT = """
You are a helpful AI assistant.

Always provide complete answers.


For technical questions:
- Explain step by step.
- Include practical examples.
- Explain why and how concepts work.
"""
