"""One SQLite file holds everything: documents, chunks, an FTS5 index and a sqlite-vec index.

* BM25 comes from SQLite's built-in FTS5 (``bm25()``), porter-stemmed.
* Vectors live in a ``vec0`` virtual table from `sqlite-vec <https://github.com/asg017/sqlite-vec>`_
  (cosine distance, exact KNN). If the extension cannot be loaded (a Python built without
  ``enable_load_extension``), the same embeddings, also stored as float32 blobs next to each
  chunk, are scanned in Python. Results are identical, only slower on big corpora.

All SQL is parameterised. The file is created ``0600`` inside a ``0700`` directory: it holds
the full text of your documents.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import struct
import threading
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path

from local_rag.chunking import Chunk
from local_rag.embeddings import STOPWORDS

log = logging.getLogger(__name__)
SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    chunks INTEGER NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    headings TEXT NOT NULL,
    page INTEGER,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    body, tokenize = 'porter unicode61 remove_diacritics 2'
);
"""


class StoreError(RuntimeError):
    """The index is missing, or was built with a different embedder."""


@dataclass(frozen=True)
class StoredChunk:
    id: int
    source: str
    title: str
    headings: tuple[str, ...]
    page: int | None
    text: str

    @property
    def breadcrumb(self) -> str:
        parts = [self.title, *[h for h in self.headings if h != self.title]]
        crumb = " > ".join(p for p in parts if p)
        return f"{crumb} (page {self.page})" if self.page is not None else crumb


def pack(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


_TOKEN = re.compile(r"\w+", re.UNICODE)


def fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: every token quoted, OR-ed together."""
    words = [t.lower() for t in _TOKEN.findall(text)]
    tokens = list(dict.fromkeys(w for w in words if w not in STOPWORDS)) or list(
        dict.fromkeys(words)
    )
    return " OR ".join(f'"{t}"' for t in tokens[:64])


def _load_sqlite_vec(conn: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
    except (AttributeError, ImportError, sqlite3.Error, OSError) as exc:
        log.info("sqlite-vec unavailable (%s); using the Python vector scan", exc)
        return False
    return True


class Store:
    def __init__(self, path: Path | str, *, use_sqlite_vec: bool = True) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not self.path.exists():
                fd = os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600)
                os.close(fd)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(_SCHEMA)
        self.vec = use_sqlite_vec and _load_sqlite_vec(self.conn)
        self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ meta
    def meta(self, key: str) -> str | None:
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def _set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def bind_embedder(self, name: str, dim: int) -> None:
        """Record (or check) which embedder built this index."""
        with self._lock:
            current, current_dim = self.meta("embedder"), self.meta("dim")
            if current is not None and (current != name or current_dim != str(dim)):
                raise StoreError(
                    f"this index was built with {current} ({current_dim}d), not {name} ({dim}d). "
                    "Re-ingest with --rebuild, or use the same embedder."
                )
            if current is None:
                self._set_meta("embedder", name)
                self._set_meta("dim", str(dim))
                self._set_meta("schema", SCHEMA_VERSION)
            if self.vec:
                self.conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING "
                    f"vec0(embedding float[{int(dim)}] distance_metric=cosine)"
                )
            self.conn.commit()

    def check_embedder(self, name: str) -> None:
        current = self.meta("embedder")
        if current is None:
            raise StoreError("the index is empty: run `local-rag ingest <folder>` first")
        if current != name:
            raise StoreError(f"this index was built with {current}; you asked for {name}")

    def clear(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM chunks_fts")
            self.conn.execute("DROP TABLE IF EXISTS chunks_vec")
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM documents")
            self.conn.execute("DELETE FROM meta")
            self.conn.commit()

    # ------------------------------------------------------------- documents
    def document_hash(self, path: str) -> str | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT sha256 FROM documents WHERE path = ?", (path,)
            ).fetchone()
        return None if row is None else str(row[0])

    def _chunk_ids(self, doc_id: int) -> list[int]:
        rows = self.conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))
        return [int(r[0]) for r in rows]

    def delete_document(self, path: str) -> None:
        with self._lock:
            row = self.conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
            if row is None:
                return
            for chunk_id in self._chunk_ids(int(row[0])):
                self.conn.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
                if self.vec:
                    self.conn.execute("DELETE FROM chunks_vec WHERE rowid = ?", (chunk_id,))
            self.conn.execute("DELETE FROM documents WHERE id = ?", (int(row[0]),))
            self.conn.commit()

    def add_document(
        self,
        *,
        path: str,
        source: str,
        kind: str,
        title: str,
        sha256: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("one embedding per chunk")
        with self._lock:
            self.delete_document(path)
            cur = self.conn.execute(
                "INSERT INTO documents(path, source, kind, title, sha256, chunks, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (path, source, kind, title, sha256, len(chunks), datetime.now(UTC).isoformat()),
            )
            doc_id = int(cur.lastrowid or 0)
            for chunk, vector in zip(chunks, embeddings, strict=True):
                blob = pack(vector)
                cur = self.conn.execute(
                    "INSERT INTO chunks(doc_id, ordinal, headings, page, text, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        doc_id,
                        chunk.ordinal,
                        json.dumps(chunk.headings),
                        chunk.page,
                        chunk.text,
                        blob,
                    ),
                )
                chunk_id = int(cur.lastrowid or 0)
                self.conn.execute(
                    "INSERT INTO chunks_fts(rowid, body) VALUES (?, ?)",
                    (chunk_id, chunk.indexed_text),
                )
                if self.vec:
                    self.conn.execute(
                        "INSERT INTO chunks_vec(rowid, embedding) VALUES (?, ?)", (chunk_id, blob)
                    )
            self.conn.commit()
            return doc_id

    def document_paths(self) -> list[tuple[str, str]]:
        with self._lock:
            rows = self.conn.execute("SELECT path, source FROM documents").fetchall()
        return [(str(p), str(s)) for p, s in rows]

    def documents(self) -> list[dict[str, object]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT source, kind, title, chunks, ingested_at FROM documents ORDER BY source"
            ).fetchall()
        keys = ("source", "kind", "title", "chunks", "ingested_at")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def stats(self) -> dict[str, object]:
        with self._lock:
            docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "documents": int(docs),
            "chunks": int(chunks),
            "embedder": self.meta("embedder"),
            "dim": int(self.meta("dim") or 0),
            "vector_index": "sqlite-vec" if self.vec else "python-scan",
        }

    # ---------------------------------------------------------------- chunks
    def get_chunks(self, ids: Iterable[int]) -> dict[int, StoredChunk]:
        wanted = list(dict.fromkeys(int(i) for i in ids))
        if not wanted:
            return {}
        with self._lock:
            rows = self.conn.execute(
                "SELECT c.id, d.source, d.title, c.headings, c.page, c.text "
                "FROM chunks c JOIN documents d ON d.id = c.doc_id "
                "WHERE c.id IN (SELECT value FROM json_each(?))",
                (json.dumps(wanted),),
            ).fetchall()
        out = {}
        for cid, source, title, headings, page, text in rows:
            out[int(cid)] = StoredChunk(
                id=int(cid),
                source=str(source),
                title=str(title),
                headings=tuple(json.loads(headings)),
                page=None if page is None else int(page),
                text=str(text),
            )
        return out

    def all_chunks(self) -> list[StoredChunk]:
        with self._lock:
            ids = [int(r[0]) for r in self.conn.execute("SELECT id FROM chunks ORDER BY id")]
        found = self.get_chunks(ids)
        return [found[i] for i in ids]

    # ---------------------------------------------------------------- search
    def search_bm25(self, query: str, limit: int) -> list[tuple[int, float]]:
        """``(chunk_id, score)`` best first; score is ``-bm25`` (higher is better)."""
        match = fts_query(query)
        if not match:
            return []
        with self._lock:
            rows = self.conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (match, int(limit)),
            ).fetchall()
        return [(int(r[0]), -float(r[1])) for r in rows]

    def search_vector(self, vector: Sequence[float], limit: int) -> list[tuple[int, float]]:
        """``(chunk_id, cosine_similarity)`` best first."""
        if self.meta("dim") is None:
            return []
        if self.vec:
            with self._lock:
                rows = self.conn.execute(
                    "SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
                    "ORDER BY distance",
                    (pack(vector), min(int(limit), 4096)),
                ).fetchall()
            return [(int(r[0]), 1.0 - float(r[1])) for r in rows]
        with self._lock:
            all_rows = self.conn.execute("SELECT id, embedding FROM chunks").fetchall()
        scored = []
        for cid, blob in all_rows:
            other = unpack(blob)
            scored.append((int(cid), sum(a * b for a, b in zip(vector, other, strict=False))))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[: int(limit)]
