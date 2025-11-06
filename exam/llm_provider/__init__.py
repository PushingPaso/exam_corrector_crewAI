"""
LLM Provider per CrewAI.
CrewAI gestisce i modelli LLM internamente negli agenti,
ma manteniamo questa configurazione per coerenza.
"""
import getpass
import os
from langchain_groq import ChatGroq

KEY_GROQ_API_KEY = "GROQ_API_KEY"


def ensure_groq_api_key():
    """Ensures Groq API key is available in environment."""
    if not os.environ.get(KEY_GROQ_API_KEY):
        os.environ[KEY_GROQ_API_KEY] = getpass.getpass("Enter API key for Groq: ")
    return os.environ[KEY_GROQ_API_KEY]


def get_llm_config(model_name: str = "llama-3.3-70b-versatile"):
    """
    Ottiene la configurazione LLM per CrewAI.

    Args:
        model_name: Nome del modello (default: llama-3.3-70b-versatile)

    Returns:
        Dict con configurazione per CrewAI Agent
    """
    # Model configurations
    model_configs = {
        "llama-3.3": "llama-3.3-70b-versatile",
        "llama-8b": "llama-3.1-8b-instant",
        "llama-4": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "openAI": "openai/gpt-oss-120b",
        "gwen": "qwen/qwen3-32b"
    }

    # Use model_name directly if it's a full Groq model name
    if model_name in model_configs:
        model_name = model_configs[model_name]
    elif not model_name:
        model_name = "llama-3.3-70b-versatile"

    ensure_groq_api_key()

    llm = ChatGroq(
        model=model_name,
        groq_api_key=os.environ[KEY_GROQ_API_KEY],
        temperature=0.1,
        max_tokens=8000,
    )

    return {
        "llm": llm,
        "model_name": model_name,
        "provider": "groq"
    }


def get_llm(model_name: str = "llama-3.3-70b-versatile"):
    """
    Ottiene l'istanza LLM diretta.
    """
    config = get_llm_config(model_name)
    return config["llm"]