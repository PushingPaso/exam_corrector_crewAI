import argparse
import shutil

from exam.rag import *


def main():
    parser = argparse.ArgumentParser(description='RAG Vector Store Manager con FAISS')
    parser.add_argument('--fill', action='store_true',
                        help='Riempie il vector store con le slide del corso')
    parser.add_argument('--dimension', type=int, default=1000,
                        help='Dimensionalità degli embeddings (default: 1000)')
    parser.add_argument('--force', action='store_true',
                        help='Forza la ricreazione del database (elimina quello esistente)')
    parser.add_argument('--model', type=str, default='faiss',
                        help='Modello (mantenuto per compatibilità, usa sempre FAISS)')

    args = parser.parse_args()

    if args.fill:
        print(f"# Creazione vector store FAISS con {args.dimension} dimensioni")

        # Forza ricreazione se richiesto
        if args.force:
            if os.path.exists(DIR_RAG_DB):
                print(f"# Eliminazione database esistente: {DIR_RAG_DB}")
                shutil.rmtree(DIR_RAG_DB)

        # Crea vector store
        vector_store = sqlite_vector_store(dimension=args.dimension)

        print(f"# Vector store creato in {DIR_RAG_DB}")
        print(f"# Riempimento con le slide del corso...")

        slide_count = 0
        batch_texts = []
        batch_metadatas = []

        for slide in all_slides():
            batch_texts.append(slide.content)
            batch_metadatas.append({
                "source": slide.source,
                "lines": slide.lines,
                "index": slide.index
            })
            slide_count += 1

            # Aggiungi in batch di 50
            if len(batch_texts) >= 50:
                vector_store.add_texts(batch_texts, batch_metadatas)
                print(f"# Aggiunte {slide_count} slide...")
                batch_texts = []
                batch_metadatas = []

        # Aggiungi eventuali rimanenti
        if batch_texts:
            vector_store.add_texts(batch_texts, batch_metadatas)

        print(f"# Vector store riempito con successo: {slide_count} slide")
        print(f"# Dimensionalità: {vector_store.get_dimensionality()}")
        print(f"# Documenti totali: {vector_store.get_collection_size()}")

    else:
        # Modalità query
        vector_store = sqlite_vector_store(dimension=args.dimension)

        print(f"# Vector store caricato con successo")
        print(f"# Dimensionalità: {vector_store.get_dimensionality()}")
        print(f"# Documenti: {vector_store.get_collection_size()}")
        print(f"\n# Inserisci query (oppure 'exit' per uscire):\n")

        while True:
            try:
                query = input("\t> ")
                if query.strip().lower() == 'exit':
                    break

                results = vector_store.similarity_search(query, k=3)

                print(f"\n\t# Trovati {len(results)} risultati:\n")
                for i, doc in enumerate(results, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"\t# Risultato {i} - Fonte: {source}")

                    # Mostra preview del contenuto
                    content = doc.page_content[:200].replace("\n", "\n\t\t")
                    print(f"\t\t{content}")
                    if len(doc.page_content) > 200:
                        print("\t\t...")
                    print("\t\t---\n")

            except (EOFError, KeyboardInterrupt):
                print("\n")
                break

    print("# Arrivederci!")


if __name__ == "__main__":
    main()