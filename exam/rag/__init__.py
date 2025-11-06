"""
Modulo RAG (Retrieval-Augmented Generation)
Tradotto per CrewAI senza dipendenze da LangChain.

SOSTITUZIONE: Abbandono di 'sqlite-vec' (problematico) in favore di 'chromadb'.
'chromadb' è una libreria standard per la persistenza di vector store.
"""
import re
import shutil
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Import nativi (NO LANGCHAIN)
from sentence_transformers import SentenceTransformer
import chromadb # Importa la nuova libreria

from exam import DIR_ROOT


DIR_CONTENT = DIR_ROOT / "content"
# MODIFICA: Chroma salva in una cartella, non in un file.
# Rinomino la variabile per chiarezza.
DIR_RAG_DB = DIR_ROOT / "slides_rag_db"
# Esporta la nuova variabile per coerenza con __main__.py
FILE_DB = DIR_RAG_DB

MARKDOWN_FILES = list(DIR_CONTENT.glob("**/_index.md"))
REGEX_SLIDE_DELIMITER = re.compile(r"^\s*(---|\+\+\+)")


class Slide(BaseModel):
    content: str
    source: str
    lines: tuple[int, int]
    index: int

    @property
    def lines_count(self):
        return self.content.count("\n") + 1 if self.content else 0


class Document(BaseModel):


    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# CARICAMENTO SLIDE (invariato)
# ============================================================================

def all_slides(files = None):
    if files is None:
        files = MARKDOWN_FILES
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            slide_beginning_line_num = 0
            line_number = 0
            slide_lines = []
            slide_index = 0
            last_was_blank = False
            for line in f.readlines():
                line_number += 1
                if REGEX_SLIDE_DELIMITER.match(line):
                    if slide_lines:
                        yield Slide(
                            content="\n".join(slide_lines),
                            source=str(file.relative_to(DIR_CONTENT)),
                            lines=(slide_beginning_line_num, line_number - 1),
                            index=slide_index,
                        )
                        slide_index += 1
                    slide_lines = []
                    slide_beginning_line_num = line_number + 1
                else:
                    if (stripped := line.strip()) or not last_was_blank:
                        slide_lines.append(line.rstrip())
                    last_was_blank = not stripped
            yield Slide(
                content="\n".join(slide_lines),
                source=str(file.relative_to(DIR_CONTENT)),
                lines=(slide_beginning_line_num, line_number - 1),
                index=slide_index,
            )


# ============================================================================
# GESTIONE EMBEDDING (con sentence_transformers, invariato)
# ============================================================================

def huggingface_embeddings(model=None) -> SentenceTransformer:
    """
    Crea un modello di embeddings HuggingFace usando sentence_transformers.
    NON USA LANGCHAIN.

    Args:
        model: Hint per il modello (es. 'bge-large')

    Returns:
        Istanza di SentenceTransformer
    """
    if not model:
        model = "bge-large"

    model = model.lower()
    device = 'cpu'

    if model == "bge-large" or model == "best":
        model_name = "BAAI/bge-large-en-v1.5"
    elif model == "bge-base" or model == "recommended":
        model_name = "BAAI/bge-base-en-v1.5"
    elif model == "bge-small" or model == "fast":
        model_name = "BAAI/bge-small-en-v1.5"
    elif model == "nomic":
        model_name = "nomic-ai/nomic-embed-text-v1"
    elif model == "gte-large":
        model_name = "thenlper/gte-large"
    elif model == "legacy-small" or "mini" in model:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    elif model == "legacy-large" or "mpnet" in model:
        model_name = "sentence-transformers/all-mpnet-base-v2"
    elif model.startswith("BAAI/") or model.startswith("sentence-transformers/") or "/" in model:
        model_name = model
    else:
        raise ValueError(f"Unknown model hint: {model}")

    print(f"# Loading embeddings model: {model_name} (via sentence-transformers)")

    return SentenceTransformer(
        model_name_or_path=model_name,
        device=device,
        trust_remote_code=True if model == "nomic" else False
    )


# ============================================================================
# WRAPPER VECTOR STORE (ora usa ChromaDB)
# ============================================================================

class CrewAIVectorStore:
    """
    Wrapper per 'chromadb.Collection' che mima l'interfaccia
    del VectorStore di LangChain per compatibilità.
    """

    def __init__(self, db_path: str, table_name: str, embed_model: SentenceTransformer):
        self._embed_model = embed_model

        # Inizializza il client persistente di Chroma
        # Salverà i dati nella cartella specificata
        self._client = chromadb.PersistentClient(path=db_path)

        # Crea un "embedding function" wrapper per Chroma
        class ChromaEmbeddingFunction(chromadb.EmbeddingFunction):
            def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
                return embed_model.encode(input, normalize_embeddings=True).tolist()

        # Ottieni o crea la "collection" (simile a una tabella)
        self._collection = self._client.get_or_create_collection(
            name=table_name,
            embedding_function=ChromaEmbeddingFunction()
        )

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Esegue la ricerca e formatta i risultati come oggetti Document compatibili.
        """
        results = []
        try:
            # Esegui la query
            query_results = self._collection.query(
                query_texts=[query],
                n_results=k
            )

            # Estrai e formatta i risultati
            docs_list = query_results.get('documents', [[]])[0]
            metas_list = query_results.get('metadatas', [[]])[0]

            for i, text_content in enumerate(docs_list):
                metadata = metas_list[i] if i < len(metas_list) else {}
                results.append(Document(
                    page_content=text_content,
                    metadata=metadata
                ))

        except Exception as e:
            print(f"Error during RAG search (ChromaDB): {e}")
            return [Document(page_content=f"Error during search: {e}", metadata={})]

        return results

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Aggiunge testi al vector store.
        Chroma richiede ID univoci, li generiamo.
        """
        # Genera ID univoci
        ids = [f"doc_{hash(txt)}_{i}" for i, txt in enumerate(texts)]

        try:
            self._collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        except chromadb.errors.IDAlreadyExistsError:
            print("Warning: Some documents were already present and were skipped.")
            # Gestisci l'aggiunta incrementale se necessario (qui usiamo 'upsert')
            self._collection.upsert(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Error adding texts to ChromaDB: {e}")


    def get_dimensionality(self) -> int:
        """
        Restituisce la dimensionalità richiesta dal resto del codice.
        La otteniamo direttamente dal modello di embedding.
        """
        try:
            dim = self._embed_model.get_sentence_embedding_dimension()
            if dim:
                return dim
            # Fallback
            return len(self._embed_model.encode("test", normalize_embeddings=True))
        except Exception:
            return 0 # Errore

# ============================================================================
# FUNZIONE PRINCIPALE (Factory)
# ============================================================================

def sqlite_vector_store(
        db_file: str = str(FILE_DB), # Mantiene il nome argomento per compatibilità
        model: str = None,
        table_name: str = "se_slides") -> CrewAIVectorStore:
    """
    Crea o carica un vector store persistente usando ChromaDB.
    MANTIENE il nome 'sqlite_vector_store' per compatibilità con il
    resto del codice, anche se ora usa ChromaDB.
    """
    # 1. Ottieni il modello SentenceTransformer
    embeddings_model = huggingface_embeddings(model)

    # 2. Crea e restituisci il wrapper
    #    Nota: db_file (che ora è DIR_RAG_DB) è una cartella
    return CrewAIVectorStore(
        db_path=db_file,
        table_name=table_name,
        embed_model=embeddings_model,
    )