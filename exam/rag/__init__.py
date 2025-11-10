"""
Modulo RAG (Retrieval-Augmented Generation)
Versione con SQLite unico file + FAISS in-memory

USO: SQLite per storage + TF-IDF embeddings a 1000 dimensioni
TUTTO IN UN SINGOLO FILE .db
"""
import re
import os
import sqlite3
import pickle
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from exam import DIR_ROOT

DIR_CONTENT = DIR_ROOT / "content"
DIR_RAG_DB = DIR_ROOT / "slides_rag_db"
FILE_DB = DIR_RAG_DB / "vector_store.db"

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
# CARICAMENTO SLIDE
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
# TF-IDF EMBEDDER (1000 dimensioni)
# ============================================================================

class TfidfEmbedder:
    """
    Embedder basato su TF-IDF + SVD per riduzione a 1000 dimensioni.
    Completamente deterministico e stabile.
    """

    def __init__(self, dimension: int = 1000):
        self.dimension = dimension
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents='unicode'
        )
        self.svd = TruncatedSVD(n_components=min(dimension, 1000))
        self.is_fitted = False

    def fit(self, texts: List[str]):
        """Addestra il vectorizer sui testi."""
        if not texts:
            raise ValueError("Cannot fit on empty text list")

        print(f"🔧 Training TF-IDF vectorizer su {len(texts)} documenti...")

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"   Vocabolario: {len(self.vectorizer.vocabulary_)} parole")
        print(f"   Matrice TF-IDF: {tfidf_matrix.shape}")

        if tfidf_matrix.shape[1] > self.dimension:
            print(f"🔧 Riduzione dimensionale a {self.dimension} dimensioni...")
            self.svd.fit(tfidf_matrix)
            explained_var = sum(self.svd.explained_variance_ratio_) * 100
            print(f"   Varianza spiegata: {explained_var:.2f}%")

        self.is_fitted = True
        print(f"✅ Embedder pronto!")

    def _ensure_dimension(self, embedding: np.ndarray) -> np.ndarray:
        """Assicura che l'embedding abbia esattamente self.dimension dimensioni."""
        current_dim = embedding.shape[0]

        if current_dim == self.dimension:
            return embedding
        elif current_dim > self.dimension:
            return embedding[:self.dimension]
        else:
            padded = np.zeros(self.dimension)
            padded[:current_dim] = embedding
            return padded

    def embed(self, text: str) -> np.ndarray:
        """Genera embedding a 1000 dimensioni per un singolo testo."""
        if not self.is_fitted:
            print("⚠️  Embedder non addestrato, uso embedding casuale")
            return np.random.randn(self.dimension).astype('float32')

        tfidf_vec = self.vectorizer.transform([text])

        if tfidf_vec.shape[1] > self.dimension:
            embedding = self.svd.transform(tfidf_vec)[0]
        else:
            embedding = tfidf_vec.toarray()[0]

        embedding = self._ensure_dimension(embedding)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype('float32')

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings per un batch di testi."""
        if not self.is_fitted:
            print("⚠️  Embedder non addestrato, uso embeddings casuali")
            return np.random.randn(len(texts), self.dimension).astype('float32')

        tfidf_matrix = self.vectorizer.transform(texts)

        if tfidf_matrix.shape[1] > self.dimension:
            embeddings = self.svd.transform(tfidf_matrix)
        else:
            embeddings = tfidf_matrix.toarray()

        result = np.zeros((len(texts), self.dimension), dtype='float32')
        for i, emb in enumerate(embeddings):
            result[i] = self._ensure_dimension(emb)

        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1
        result = result / norms

        return result

# ============================================================================
# VECTOR STORE CON SQLITE UNICO FILE
# ============================================================================

class CrewAIVectorStore:
    """
    Vector Store che usa SQLite per storage (tutto in un file .db).
    FAISS viene ricostruito in memoria al caricamento.
    """

    def __init__(self, db_file: str, dimension: int = 1000):
        """
        Args:
            db_file: Percorso del file .db SQLite
            dimension: Dimensionalità degli embeddings (default: 1000)
        """
        self.dimension = dimension
        self.db_file = db_file

        # Crea directory se non esiste
        os.makedirs(os.path.dirname(db_file) if os.path.dirname(db_file) else ".", exist_ok=True)

        # Connessione SQLite
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self._init_database()

        # Inizializza embedder
        self.embedder = TfidfEmbedder(dimension=dimension)

        # Carica vectorizer se esiste
        self._load_embedder()

        # Index FAISS (in memoria)
        self.index = faiss.IndexFlatIP(self.dimension)

        # Carica vettori esistenti
        self._load_vectors()

    def _init_database(self):
        """Inizializza le tabelle SQLite."""
        cursor = self.conn.cursor()

        # Tabella per documenti e metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        ''')

        # Tabella per vettori
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                doc_id INTEGER PRIMARY KEY,
                vector BLOB NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id)
            )
        ''')

        # Tabella per embedder (vectorizer + svd)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embedder_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                vectorizer BLOB,
                svd BLOB,
                is_fitted INTEGER DEFAULT 0
            )
        ''')

        self.conn.commit()
        print(f"✅ Database SQLite inizializzato: {self.db_file}")

    def _load_embedder(self):
        """Carica lo stato dell'embedder dal database."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT vectorizer, svd, is_fitted FROM embedder_state WHERE id = 1')
        row = cursor.fetchone()

        if row and row[2]:  # is_fitted
            try:
                self.embedder.vectorizer = pickle.loads(row[0])
                self.embedder.svd = pickle.loads(row[1])
                self.embedder.is_fitted = True
                print(f"✅ Caricato embedder pre-addestrato")
            except Exception as e:
                print(f"⚠️  Errore caricamento embedder: {e}")

    def _save_embedder(self):
        """Salva lo stato dell'embedder nel database."""
        if not self.embedder.is_fitted:
            return

        cursor = self.conn.cursor()

        vectorizer_blob = pickle.dumps(self.embedder.vectorizer)
        svd_blob = pickle.dumps(self.embedder.svd)

        cursor.execute('''
            INSERT OR REPLACE INTO embedder_state (id, vectorizer, svd, is_fitted)
            VALUES (1, ?, ?, 1)
        ''', (vectorizer_blob, svd_blob))

        self.conn.commit()
        print(f"💾 Salvato embedder")

    def _load_vectors(self):
        """Carica tutti i vettori nel FAISS index in memoria."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT vector FROM vectors ORDER BY doc_id')

        vectors = []
        for row in cursor:
            vector = np.frombuffer(row[0], dtype='float32')
            vectors.append(vector)

        if vectors:
            vectors_array = np.array(vectors)
            self.index.add(vectors_array)
            print(f" Caricati {len(vectors)} vettori in memoria")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Esegue ricerca di similarità."""
        try:
            if self.index.ntotal == 0:
                return [Document(page_content="Vector store vuoto", metadata={})]

            # Genera embedding per la query
            query_embedding = self.embedder.embed(query)
            query_embedding = query_embedding.reshape(1, -1)

            # Normalizza
            faiss.normalize_L2(query_embedding)

            # Cerca
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, k)

            # Recupera documenti dal database
            results = []
            cursor = self.conn.cursor()

            for idx in indices[0]:
                if idx >= 0:
                    doc_id = idx + 1  # IDs in SQLite partono da 1
                    cursor.execute(
                        'SELECT content, metadata FROM documents WHERE id = ?',
                        (doc_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        results.append(Document(
                            page_content=row[0],
                            metadata=pickle.loads(row[1])
                        ))

            return results

        except Exception as e:
            print(f" Errore nella ricerca: {e}")
            import traceback
            traceback.print_exc()
            return [Document(page_content="Errore nella ricerca", metadata={})]

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Aggiunge testi al vector store."""
        if not texts:
            return

        if metadatas is None:
            metadatas = [{} for _ in texts]

        try:
            cursor = self.conn.cursor()

            # Se embedder non addestrato, addestralo
            if not self.embedder.is_fitted:
                print(f"🔧 Primo batch: addestramento vectorizer...")

                # Carica tutti i documenti esistenti
                cursor.execute('SELECT content FROM documents')
                existing_texts = [row[0] for row in cursor.fetchall()]
                all_texts = existing_texts + texts

                self.embedder.fit(all_texts)
                self._save_embedder()

                # Se ci sono documenti esistenti, rigenera embeddings
                if existing_texts:
                    print(f"Rigenerazione embeddings per {len(existing_texts)} documenti...")
                    old_embeddings = self.embedder.embed_batch(existing_texts)
                    faiss.normalize_L2(old_embeddings)
                    self.index.reset()
                    self.index.add(old_embeddings)

                    # Aggiorna vettori nel database
                    for i, emb in enumerate(old_embeddings):
                        cursor.execute(
                            'UPDATE vectors SET vector = ? WHERE doc_id = ?',
                            (emb.tobytes(), i + 1)
                        )

            # Genera embeddings per nuovi testi
            embeddings = self.embedder.embed_batch(texts)
            faiss.normalize_L2(embeddings)

            # Aggiungi documenti al database
            for text, metadata, embedding in zip(texts, metadatas, embeddings):
                # Inserisci documento
                cursor.execute(
                    'INSERT INTO documents (content, metadata) VALUES (?, ?)',
                    (text, pickle.dumps(metadata))
                )
                doc_id = cursor.lastrowid

                # Inserisci vettore
                cursor.execute(
                    'INSERT INTO vectors (doc_id, vector) VALUES (?, ?)',
                    (doc_id, embedding.tobytes())
                )

            # Aggiungi a FAISS in memoria
            self.index.add(embeddings)

            self.conn.commit()

            print(f"➕ Aggiunti {len(texts)} documenti (totale: {self.index.ntotal})")

        except Exception as e:
            self.conn.rollback()
            print(f" Errore nell'aggiunta: {e}")
            import traceback
            traceback.print_exc()

    def get_dimensionality(self) -> int:
        """Restituisce la dimensionalità."""
        return self.dimension

    def get_collection_size(self) -> int:
        """Restituisce il numero di documenti."""
        return self.index.ntotal

    def close(self):
        """Chiude la connessione al database."""
        if self.conn:
            self.conn.close()
            print("Database chiuso")

    def __del__(self):
        """Chiude la connessione quando l'oggetto viene distrutto."""
        self.close()


def sqlite_vector_store(
        db_file: str = str(FILE_DB),
        model: str = None,
        table_name: str = "se_slides",
        dimension: int = 1000) -> CrewAIVectorStore:
    """
    Crea o carica un vector store con SQLite (singolo file .db).

    Args:
        db_file: Percorso del file .db
        model: Non usato (per compatibilità)
        table_name: Non usato (per compatibilità)
        dimension: Dimensionalità degli embeddings (default: 1000)
    """
    try:
        store = CrewAIVectorStore(db_file=db_file, dimension=dimension)
        print(f" Vector store inizialize: {db_file}")
        print(f"Dimensionality: {dimension}")
        print(f" Documents: {store.get_collection_size()}")
        return store
    except Exception as e:
        print(f" Critic Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def populate_vector_store(store: CrewAIVectorStore, max_slides: int = None) -> int:
    """Popola il vector store con le slide."""
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

            if len(texts) >= 50:
                store.add_texts(texts, metadatas)
                texts.clear()
                metadatas.clear()

        if texts:
            store.add_texts(texts, metadatas)

        print(f"Aggiunte {slides_added} slide al vector store")
        return slides_added

    except Exception as e:
        print(f"Errore nel popolamento: {e}")
        import traceback
        traceback.print_exc()
        return slides_added

def search_slides(query: str, store: CrewAIVectorStore, k: int = 5) -> List[Document]:
    """Helper per cercare slide."""
    try:
        return store.similarity_search(query, k)
    except Exception as e:
        print(f" Errore nella ricerca: {e}")
        return []