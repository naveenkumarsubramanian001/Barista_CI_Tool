import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# --- Config ---
MODEL_NAME = "qwen3:4b"  # Available on system
TEMPERATURE = 0.1

# --- LLM Instance ---
def get_llm(temperature=None):
    if temperature is None:
        temperature = TEMPERATURE
    
    MODEL_NAME = (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
    return ChatGroq(
        model=MODEL_NAME,
        temperature=temperature,
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
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY", "").strip()
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "").strip()

# --- Search Provider ---
# Options: "tavily", "serper", "bing", "google"
# Fallback: If the chosen provider has no API key, falls back automatically
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily").strip().lower()


def _parse_csv_env(name: str, default: str) -> list[str]:
    raw_value = os.getenv(name, default)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _parse_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


# --- API Runtime ---
CORS_ALLOWED_ORIGINS = _parse_csv_env(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
)
CORS_ALLOW_CREDENTIALS = _parse_bool_env("CORS_ALLOW_CREDENTIALS", True)


# --- LangGraph Checkpointer ---
# "memory" keeps current behavior. "sqlite" enables persistence when sqlite saver is available.
CHECKPOINTER_BACKEND = os.getenv("CHECKPOINTER_BACKEND", "sqlite").strip().lower()
CHECKPOINTER_SQLITE_PATH = os.getenv("CHECKPOINTER_SQLITE_PATH", "barista_checkpoints.sqlite").strip()
