import getpass
import os
from crewai import LLM

KEY_GROQ_API_KEY = "GROQ_API_KEY"


def ensure_groq_api_key():
    """Ensures Groq API key is available in environment."""
    if not os.environ.get(KEY_GROQ_API_KEY):
        os.environ[KEY_GROQ_API_KEY] = getpass.getpass("Enter API key for Groq: ")
    return os.environ[KEY_GROQ_API_KEY]


def get_llm(model_name: str = None) -> LLM:
    """
    Configura e restituisce un *dizionario di configurazione* (llm_config)
    per un modello Groq specifico, da passare agli agenti CrewAI.

    Args:
        model_name (str): Il nome del modello da usare.
                          Accetta shortcut come "llama-3.3", "llama-8b", o "llama-inf".

    Returns:
        dict: Un dizionario di configurazione per CrewAI.
    """

    # Dizionario per gli shortcut dei modelli
    model_configs = {
        "llama-3.3": "llama-3.3-70b-versatile",
        "llama-8b": "llama-3.1-8b-instant",
        "llama-inf": "meta-llama/llama-4-scout-17b-16e-instruct",
    }

    # 1. Gestisci il default
    if model_name is None:
        #model_name = "llama-3.3-70b-versatile"  # Questo è lo shortcut
        model_name = "openai/gpt-4o"

    # 3. Crea il dizionario di configurazione
    #    Usa una f-string (la 'f' all'inizio) per inserire la variabile!
    llm = LLM(
        model=model_name,
        stream=True  # Enable streaming
    )

    # 4. Restituisci il dizionario di configurazione
    return llm


class AIOracle:
    """Base class for AI-powered operations using Groq."""

    def __init__(self, model_name: str = None):
        self.__llm = get_llm(model_name)

    @property
    def llm(self):
        return self.__llm

    @property
    def model_name(self):
        return "llama-3.3-70b-versatile"

    @property
    def model_provider(self):
        return "groq"