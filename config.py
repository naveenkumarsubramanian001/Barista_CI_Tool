import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

# --- Config ---
MODEL_NAME = "qwen3:4b"  # As requested across Phase 2
TEMPERATURE = 0.1

# --- LLM Instance ---
def get_llm():
    return ChatOllama(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        format="json", # Force JSON output if possible
    )

# --- API Keys ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
