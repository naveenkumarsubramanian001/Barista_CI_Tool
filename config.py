import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# --- Config ---
TEMPERATURE = 0.1


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_env(name: str, default: str) -> list[str]:
    raw_value = os.getenv(name, default)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


# --- LLM Instance ---
def get_llm(temperature=None):
    if temperature is None:
        temperature = TEMPERATURE

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured. Please set it in .env.")

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
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
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
GROQ_MODEL = (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BING_SEARCH_API_KEY = (os.getenv("BING_SEARCH_API_KEY") or "").strip()
SERPER_API_KEY = (os.getenv("SERPER_API_KEY") or "").strip()
GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
GOOGLE_CSE_ID = (os.getenv("GOOGLE_CSE_ID") or "").strip()

# --- Search Provider / Strategy ---
# SEARCH_PROVIDER options: "tavily", "serper", "bing", "google"
# SEARCH_STRATEGY options: "parallel", "single", "fallback"
SEARCH_PROVIDER = (os.getenv("SEARCH_PROVIDER") or "tavily").strip().lower()
SEARCH_STRATEGY = (os.getenv("SEARCH_STRATEGY") or "parallel").strip().lower()

# --- CORS ---
CORS_ALLOWED_ORIGINS = _parse_csv_env(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
)
CORS_ALLOW_CREDENTIALS = _env_flag("CORS_ALLOW_CREDENTIALS", default=True)

# --- Checkpoint Persistence ---
CHECKPOINTER_BACKEND = (os.getenv("CHECKPOINTER_BACKEND") or "sqlite").strip().lower()
# Keep both names for compatibility across modules.
CHECKPOINTER_SQLITE_PATH = (
    os.getenv("CHECKPOINTER_SQLITE_PATH")
    or os.getenv("CHECKPOINT_DB_PATH")
    or "barista_checkpoints.sqlite"
).strip()
CHECKPOINT_DB_PATH = CHECKPOINTER_SQLITE_PATH
STRICT_STARTUP_VALIDATION = _env_flag("STRICT_STARTUP_VALIDATION", default=False)

# --- Discriminator Feature Flags ---
# Keep disabled by default to preserve stable scoring behavior.
ENABLE_FUZZY_SCORING = _env_flag("ENABLE_FUZZY_SCORING", default=False)


def provider_readiness() -> Dict[str, bool]:
    """Return per-provider readiness based on configured API keys."""
    return {
        "tavily": bool(TAVILY_API_KEY),
        "serper": bool(SERPER_API_KEY),
        "google": bool(GOOGLE_API_KEY and GOOGLE_CSE_ID),
        "bing": bool(BING_SEARCH_API_KEY),
    }


def available_providers() -> List[str]:
    """Return providers that are currently configured and usable."""
    ready = provider_readiness()
    return [name for name, ok in ready.items() if ok]


def validate_runtime_config(strict: bool | None = None) -> Dict[str, object]:
    """Validate runtime settings.

    Returns a report and optionally raises RuntimeError when strict mode is on.
    """
    if strict is None:
        strict = STRICT_STARTUP_VALIDATION

    errors: List[str] = []
    warnings: List[str] = []

    if SEARCH_STRATEGY not in {"parallel", "single", "fallback"}:
        errors.append(f"Invalid SEARCH_STRATEGY='{SEARCH_STRATEGY}'. Use parallel|single|fallback.")

    if SEARCH_PROVIDER not in {"tavily", "serper", "google", "bing"}:
        errors.append(f"Invalid SEARCH_PROVIDER='{SEARCH_PROVIDER}'. Use tavily|serper|google|bing.")

    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY is not configured. LLM calls will fail.")

    providers = available_providers()
    if not providers:
        errors.append("No search providers are configured. Set at least one provider API key.")

    if SEARCH_STRATEGY in {"single", "fallback"} and SEARCH_PROVIDER not in providers:
        warnings.append(
            f"SEARCH_PROVIDER='{SEARCH_PROVIDER}' is not configured; strategy '{SEARCH_STRATEGY}' will degrade to available providers: {providers}."
        )

    report: Dict[str, object] = {
        "providers": provider_readiness(),
        "available_providers": providers,
        "search_strategy": SEARCH_STRATEGY,
        "search_provider": SEARCH_PROVIDER,
        "errors": errors,
        "warnings": warnings,
    }

    if strict and errors:
        raise RuntimeError("Startup configuration validation failed: " + " | ".join(errors))

    return report
