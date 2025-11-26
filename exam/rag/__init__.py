import sqlite3
import json
import re
import time
from pathlib import Path
from pydantic import BaseModel

# --- NUOVI IMPORT NATIVI ---
from openai import OpenAI
import sqlite_vec
from exam.llm_provider import ensure_openai_api_key
from exam import DIR_ROOT

# --- COSTANTI ---
DIR_CONTENT = DIR_ROOT / "content"
FILE_DB = DIR_ROOT / "slides-rag.db"
MARKDOWN_FILES = list(DIR_CONTENT.glob("**/_index.md"))
REGEX_SLIDE_DELIMITER = re.compile(r"^\s*(---|\+\+\+)")


# --- 1. MODELLO DATI E PARSER (Il tuo codice originale) ---

class Slide(BaseModel):
    content: str
    source: str
    lines: tuple[int, int]
    index: int

    @property
    def lines_count(self):
        return self.content.count("\n") + 1 if self.content else 0


def all_slides(files=None):
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


# --- 2. LOGICA NATIVA (Sostituzione LangChain) ---

def get_model_config(model_hint):
    """Restituisce nome modello e dimensione vettore."""
    model_hint = model_hint.lower() if model_hint else "small"

    if "large" in model_hint:
        return "text-embedding-3-large", 3072
    elif "old" in model_hint or "ada" in model_hint:
        return "text-embedding-ada-002", 1536
    else:
        # Default: small
        return "text-embedding-3-small", 1536


class NativeVectorStore:
    """
    Classe wrapper che gestisce SQLite + sqlite-vec + OpenAI
    Sostituisce le astrazioni di LangChain.
    """

    def __init__(self, db_file: str, model_name: str, dims: int, table_name: str = "se_slides"):
        self.db_file = db_file
        self.model_name = model_name
        self.dims = dims
        self.table_meta = f"{table_name}_meta"
        self.table_vec = f"{table_name}_vec"

        self.client = OpenAI()

        # Connessione persistente
        self.conn = sqlite3.connect(db_file)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        # Inizializza tabelle se non esistono
        self._init_db()

    def _init_db(self):
        with self.conn:
            # 1. Tabella Metadati (Testo)
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_meta} (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    source TEXT,
                    lines TEXT,
                    slide_index INTEGER
                )
            """)
            # 2. Tabella Vettoriale (Virtuale)
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_vec} USING vec0(
                    rowid INTEGER PRIMARY KEY,
                    embedding FLOAT[{self.dims}]
                )
            """)

    def _get_embedding(self, text: str):
        # Pulisce input e chiama OpenAI
        text = text.replace("\n", " ").strip()
        if not text: return None
        return self.client.embeddings.create(input=[text], model=self.model_name).data[0].embedding

    def add_slide(self, slide: Slide):
        """Genera embedding e salva nel DB."""
        # Check preventivo stringa vuota
        clean_content = slide.content.strip()
        if not clean_content:
            return False  # Skip

        vector = self._get_embedding(clean_content)
        if not vector:
            return False

        with self.conn:
            # Inserimento Metadata
            cur = self.conn.execute(f"""
                INSERT INTO {self.table_meta} (content, source, lines, slide_index)
                VALUES (?, ?, ?, ?)
            """, (slide.content, slide.source, json.dumps(slide.lines), slide.index))

            row_id = cur.lastrowid

            # Inserimento Vettore (Binary serialization)
            self.conn.execute(f"""
                INSERT INTO {self.table_vec}(rowid, embedding)
                VALUES (?, ?)
            """, (row_id, sqlite_vec.serialize_float32(vector)))
        return True

    def search(self, query: str, k: int = 4):
        """
        Esegue la ricerca semantica con sqlite-vec.
        Usa la sintassi 'k = ?' richiesta da sqlite-vec per le query KNN.
        """
        query_vector = self._get_embedding(query)
        if not query_vector:
            return []

        vector_blob = sqlite_vec.serialize_float32(query_vector)

        # Sintassi corretta per sqlite-vec: usa 'k = ?' nella WHERE
        sql = f"""
            WITH matches AS (
                SELECT rowid, distance
                FROM {self.table_vec}
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT 
                meta.content, 
                meta.source, 
                meta.lines, 
                meta.slide_index,
                m.distance
            FROM matches m
            LEFT JOIN {self.table_meta} meta ON m.rowid = meta.rowid
            ORDER BY m.distance
        """

        cursor = self.conn.execute(sql, (vector_blob, k))

        results = []
        for row in cursor.fetchall():
            results.append({
                "content": row[0],
                "source": row[1],
                "lines": json.loads(row[2]) if row[2] else [],
                "index": row[3],
                "distance": row[4]
            })
        return results

    def close(self):
        self.conn.close()


def sqlite_vector_store(
        db_file: str = str(FILE_DB),
        model: str = None,
        table_name: str = "se_slides"):
    """
    Funzione helper per ottenere l'istanza del DB (simile alla tua vecchia implementazione).
    Restituisce un oggetto NativeVectorStore.
    """
    ensure_openai_api_key()
    model_name, dims = get_model_config(model)

    return NativeVectorStore(
        db_file=db_file,
        model_name=model_name,
        dims=dims,
        table_name=table_name
    )