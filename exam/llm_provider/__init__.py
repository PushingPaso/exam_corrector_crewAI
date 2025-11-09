import getpass
import os
from crewai import LLM

# Chiave per la variabile d'ambiente
KEY_GROQ_API_KEY = "GROQ_API_KEY"


def ensure_groq_api_key():
    """
    Assicura che la chiave API di Groq sia disponibile nell'ambiente
    e la restituisce.
    """
    if not os.environ.get(KEY_GROQ_API_KEY):
        os.environ[KEY_GROQ_API_KEY] = getpass.getpass("Inserisci la tua API key per Groq: ")
    # Restituiamo la chiave per poterla passare all'istanza LLM
    return os.environ.get(KEY_GROQ_API_KEY)


def get_llm(model_name: str = None, response_format=None) -> LLM:
    """
    Configura e restituisce un'istanza della classe LLM di CrewAI
    per un modello Groq specifico.

    Args:
        model_name (str): Il nome del modello da usare.
        response_format: format of the output

    Returns:
        LLM: Un'istanza della classe LLM di CrewAI.
    """

    # Dizionario per gli shortcut dei modelli
    model_configs = {
        "llama-3.3": "llama-3.3-70b-versatile",
        "llama-8b": "llama-3.1-8b-instant",
        "llama-inf": "meta-llama/llama-4-scout-17b-16e-instruct",
    }

    if model_name is None:
        model_name = "llama-3.3-70b-versatile"

    # Build LLM kwargs
    llm_kwargs = {
        "model": f"groq/{model_name}",
        "temperature": 0.1,
    }

    # Only add response_format if it's provided
    if response_format is not None:
        # For Groq, use JSON mode instead of response_format for Pydantic models
        llm_kwargs["response_format"] = {"type": "json_object"}

    llm = LLM(**llm_kwargs)

    return llm
