import getpass
import os
from crewai import LLM  # <- Importiamo la classe LLM

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

    # Risolve lo shortcut, se usato
    if model_name in model_configs:
        model_name = model_configs[model_name]

    # Assicura che la chiave API sia impostata e la ottiene
    api_key = ensure_groq_api_key()

    # Aggiunge il prefisso 'groq/' se non è già presente
    if not model_name.startswith("groq/"):
        model_name = f"groq/{model_name}"

    # --- Modifica Chiave ---
    # Crea e restituisce l'istanza della classe LLM
    # La classe LLM di CrewAI (che fa da wrapper per LiteLLM)
    # accetta il model_name e la api_key.
    llm_instance = LLM(
        model=model_name,
        api_key=api_key
    )

    return llm_instance

if __name__ == "__main__":
    print("Test della funzione get_groq_llm...")

    # 1. Ottenere l'LLM con il modello di default
    try:
        # Potrebbe chiedere la chiave API la prima volta
        default_llm = get_llm()
        print(f"\nOttenuto LLM (default):")
        print(f"  Modello: {default_llm.model_name}")
        print(f"  Classe: {type(default_llm)}")

        # 2. Ottenere l'LLM usando uno shortcut
        llm_8b = get_llm(model_name="llama-8b")
        print(f"\nOttenuto LLM (shortcut 'llama-8b'):")
        print(f"  Modello: {llm_8b.model_name}")
        print(f"  Classe: {type(llm_8b)}")

        # Ora puoi passare 'default_llm' o 'llm_8b' direttamente
        # al parametro 'llm' dei tuoi Agent in CrewAI, ad esempio:
        #
        # mio_agente = Agent(
        #     role="Analista",
        #     goal="Analizzare i dati",
        #     backstory="...",
        #     llm=default_llm  # <- Passi l'oggetto LLM
        # )

    except Exception as e:
        print(f"\nErrore durante il test: {e}")