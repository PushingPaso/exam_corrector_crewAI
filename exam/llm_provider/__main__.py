from exam.llm_provider import get_llm

def main():
    print("Test della funzione get_groq_llm...")

    # 1. Ottenere l'LLM con il modello di default
        # Potrebbe chiedere la chiave API la prima volta
    default_llm = get_llm()
    print(f"\nOttenuto LLM (default):")
    print(f"  Modello: {default_llm}")
    print(f"  Classe: {type(default_llm)}")

    # 2. Ottenere l'LLM usando uno shortcut
    llm_8b = get_llm(model_name="llama-8b")
    print(f"\nOttenuto LLM (shortcut 'llama-8b'):")
    print(f"  Modello: {llm_8b}")
    print(f"  Classe: {type(llm_8b)}")



if __name__ == "__main__":
    main()