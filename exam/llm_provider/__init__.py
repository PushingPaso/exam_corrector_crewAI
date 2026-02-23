import getpass
import os
from crewai import LLM

KEY_GROQ_API_KEY = "GROQ_API_KEY"


def ensure_groq_api_key():
    """Ensures Groq API key is available in environment."""
    if not os.environ.get(KEY_GROQ_API_KEY):
        os.environ[KEY_GROQ_API_KEY] = getpass.getpass("Enter API key for Groq: ")
    return os.environ[KEY_GROQ_API_KEY]

KEY_OPENAI_API_KEY = "OPENAI_API_KEY"


def ensure_openai_api_key():
    if not os.environ.get(KEY_OPENAI_API_KEY):
        os.environ[KEY_OPENAI_API_KEY] = getpass.getpass("Enter API key for OpenAI: ")
    return os.environ[KEY_OPENAI_API_KEY]


def get_llm(model_name: str = None) -> LLM:
    """
    Returns:
        llm model you ask for, gpt-4o is the default
    """

    model_configs = {
        "llama-3.3": "llama-3.3-70b-versatile",
        "llama-8b": "llama-3.1-8b-instant",
        "llama-inf": "meta-llama/llama-4-scout-17b-16e-instruct",
    }

    if model_name is None:
        model_name = "gpt-4o"


    llm = LLM(
    #model=f"groq/{model_name}",
    model = model_name,
    temperature=0.0,
    )

    return llm


class AIOracle:
    """Base class for AI-powered operations using Groq."""

    def __init__(self, model_name: str = None):
        self.__llm = get_llm(model_name)
        # Store the model name (handling the None case if needed, or just what was passed)
        self._model_name = model_name if model_name else "llama-3.3-70b-versatile"

    @property
    def llm(self):
        return self.__llm

    @property
    def model_name(self):
        return self._model_name

    @property
    def model_provider(self):
        return "OpenAI"