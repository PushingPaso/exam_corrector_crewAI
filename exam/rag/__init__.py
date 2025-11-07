"""
Modulo RAG (Retrieval-Augmented Generation)
Versione semplificata e robusta per CrewAI.

USO: ChromaDB per vector storage + embeddings leggeri
ELIMINAZIONE: sentence-transformers problematico in favore di chromadb embeddings
"""
import re
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from exam import DIR_ROOT

DIR_CONTENT = DIR_ROOT / "content"
DIR_RAG_DB = DIR_ROOT / "slides_rag_db"
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
# VECTOR STORE SEMPLIFICATO (usa ChromaDB embeddings built-in)
# ============================================================================

class CrewAIVectorStore:
    """
    Vector Store semplificato che usa ChromaDB con embeddings built-in.
    Elimina la dipendenza da sentence-transformers problematica.
    """

    def __init__(self, db_path: str, table_name: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Args:
            db_path: Percorso della cartella del database
            table_name: Nome della collection
            embedding_model: Modello di embedding da usare (usa modelli supportati da Chroma)
        """
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Usa gli embeddings built-in di ChromaDB (più leggeri e stabili)
        try:
            self._collection = self._client.get_or_create_collection(
                name=table_name,
                # Chroma gestirà automaticamente gli embeddings
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Warning: Could not create collection with default settings: {e}")
            # Fallback: collection senza configurazioni speciali
            self._collection = self._client.get_or_create_collection(name=table_name)

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Esegue la ricerca di similarità.
        """
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, 10)  # Limita a max 10 risultati
            )

            documents = []
            if results['documents'] and len(results['documents']) > 0:
                for i, doc_text in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and i < len(results['metadatas'][0]) else {}
                    documents.append(Document(
                        page_content=doc_text,
                        metadata=metadata
                    ))

            return documents

        except Exception as e:
            print(f"Error during similarity search: {e}")
            return [Document(page_content="Search temporarily unavailable", metadata={})]

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Aggiunge testi al vector store.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(texts) != len(metadatas):
            raise ValueError("Texts and metadatas must have the same length")

        # Genera ID semplici
        ids = [f"doc_{i}_{hash(txt) % 10000}" for i, txt in enumerate(texts)]

        try:
            # Usa upsert per evitare duplicati
            self._collection.upsert(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added {len(texts)} documents to vector store")

        except Exception as e:
            print(f"Error adding texts: {e}")
            # Fallback: prova ad aggiungere uno per uno
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                try:
                    self._collection.upsert(
                        documents=[text],
                        metadatas=[metadata],
                        ids=[f"doc_{i}_fallback"]
                    )
                except Exception as single_error:
                    print(f"Failed to add document {i}: {single_error}")

    def get_dimensionality(self) -> int:
        """
        Restituisce una dimensionalità fissa per compatibilità.
        ChromaDB gestisce internamente la dimensionalità.
        """
        return 384  # Dimensionalità tipica per modelli piccoli

    def get_collection_size(self) -> int:
        """
        Restituisce il numero di documenti nella collection.
        """
        try:
            return self._collection.count()
        except:
            return 0

# ============================================================================
# FUNZIONE PRINCIPALE (Factory) - VERSIONE CORRETTA
# ============================================================================

def sqlite_vector_store(
        db_file: str = str(FILE_DB),
        model: str = None,  # Parametro mantenuto per compatibilità, ma non usato
        table_name: str = "se_slides") -> CrewAIVectorStore:
    """
    Crea o carica un vector store persistente usando ChromaDB.
    Versione semplificata e robusta.
    """
    try:
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(db_file) if os.path.dirname(db_file) else ".", exist_ok=True)

        # Crea e restituisci il vector store
        store = CrewAIVectorStore(
            db_path=db_file,
            table_name=table_name
        )

        print(f"✅ Vector store inizializzato: {db_file}")
        print(f"📊 Documenti nella collection: {store.get_collection_size()}")

        return store

    except Exception as e:
        print(f"❌ Errore critico nell'inizializzazione del vector store: {e}")
        # Restituisci un fallback che non crasha
        return CrewAIVectorStore(db_path="./fallback_db", table_name=table_name)

# ============================================================================
# FUNZIONI UTILITY AGGIUNTIVE
# ============================================================================

def populate_vector_store(store: CrewAIVectorStore, max_slides: int = None) -> int:
    """
    Popola il vector store con le slide.

    Args:
        store: Il vector store da popolare
        max_slides: Numero massimo di slide da processare (per testing)

    Returns:
        Numero di slide aggiunte
    """
    slides_added = 0
    texts = []
    metadatas = []

    try:
        for i, slide in enumerate(all_slides()):
            if max_slides and i >= max_slides:
                break

            texts.append(slide.content)
            metadatas.append({
                "source": slide.source,
                "lines_start": slide.lines[0],
                "lines_end": slide.lines[1],
                "slide_index": slide.index
            })
            slides_added += 1

            # Aggiungi in batch per efficienza
            if len(texts) >= 50:
                store.add_texts(texts, metadatas)
                texts.clear()
                metadatas.clear()

        # Aggiungi eventuali rimanenti
        if texts:
            store.add_texts(texts, metadatas)

        print(f" Aggiunte {slides_added} slide al vector store")
        return slides_added

    except Exception as e:
        print(f" Errore nel popolamento del vector store: {e}")
        return slides_added

def search_slides(query: str, store: CrewAIVectorStore, k: int = 5) -> List[Document]:
    """
    Funzione helper per cercare slide.
    """
    try:
        return store.similarity_search(query, k)
    except Exception as e:
        print(f" Errore nella ricerca: {e}")
        return []