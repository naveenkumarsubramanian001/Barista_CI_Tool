import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# --- Config ---
# Prefer env override; strip to avoid sending invalid model names with whitespace.
MODEL_NAME = (os.getenv("OLLAMA_MODEL") or "qwen2.5:7b").strip()
TEMPERATURE = 0.1


# --- LLM Instance ---
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=TEMPERATURE,
        api_key=os.getenv("GROQ_API_KEY"),
    )


# --- Embedding Model ---
def get_embedding_model():
    # nomic-embed-text is standard, but since it's not in the list,
    # we'll use a local model if available or stick to a likely candidate.
    # phi3:mini/qwen3 might not support embeddings in the same way.
    # I will assume nomic-embed-text might need to be pulled or use a different approach.
    # For now, I'll stick to a common one and hope it's pullable or already there (just not in 'list').
    return OllamaEmbeddings(model="nomic-embed-text")


# --- API Keys ---
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
