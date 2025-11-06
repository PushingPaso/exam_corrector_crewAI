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


def get_llm(model_name: str = "llama-3.3-70b-versatile") -> LLM:
    """
    Configura e restituisce un'istanza della classe LLM di CrewAI
    per un modello Groq specifico.

    Args:
        model_name (str): Il nome del modello da usare.
                          Accetta shortcut come "llama-3.3" o "llama-8b".

    Returns:
        LLM: Un'istanza della classe LLM di CrewAI.
    """

    # Dizionario per gli shortcut dei modelli
    model_configs = {
        "llama-3.3": "llama-3.3-70b-versatile",
        "llama-8b": "llama-3.1-8b-instant",
    }
    llm = LLM(
        model = "groq/meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1
    )

    return llm
