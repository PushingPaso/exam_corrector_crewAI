from exam.llm_provider import get_llm

def main():
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

if __name__ == "__main__":
    main()