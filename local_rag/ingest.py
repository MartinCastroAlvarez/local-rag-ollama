"""Ingest: discover -> load -> chunk -> embed -> store. Incremental by content hash."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from local_rag.chunking import chunk_document
from local_rag.config import Settings
from local_rag.embeddings import Embedder
from local_rag.loaders import IngestError
from local_rag.loaders import discover
from local_rag.loaders import load
from local_rag.loaders import read_bytes
from local_rag.store import Store

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    chunks: int = 0

    def summary(self) -> str:
        return (
            f"{len(self.added)} indexed ({self.chunks} chunks), {len(self.unchanged)} unchanged, "
            f"{len(self.removed)} removed, {len(self.errors)} failed"
        )


def ingest(
    root: Path | str,
    *,
    settings: Settings,
    store: Store,
    embedder: Embedder,
    prune: bool = False,
) -> IngestReport:
    report = IngestReport()
    fingerprint = f"{embedder.name}|{settings.chunk_chars}|{settings.overlap_chars}"
    seen: set[str] = set()
    base = Path(root).expanduser().resolve(strict=True)
    for path, source in discover(root, max_bytes=settings.max_file_bytes):
        key = str(path)
        seen.add(key)
        try:
            data = read_bytes(path, settings.max_file_bytes)
            digest = hashlib.sha256(data + fingerprint.encode()).hexdigest()
            if store.document_hash(key) == digest:
                report.unchanged.append(source)
                continue
            doc = load(path, source, max_bytes=settings.max_file_bytes, data=data)
            chunks = chunk_document(
                doc, chunk_chars=settings.chunk_chars, overlap_chars=settings.overlap_chars
            )
            if not chunks:
                raise IngestError("no extractable text")
            vectors = embedder.embed_documents([c.indexed_text for c in chunks])
            store.bind_embedder(embedder.name, len(vectors[0]))
            store.add_document(
                path=key,
                source=source,
                kind=doc.kind,
                title=doc.title,
                sha256=digest,
                chunks=chunks,
                embeddings=vectors,
            )
            report.added.append(source)
            report.chunks += len(chunks)
            log.info("indexed %s: %d chunks", source, len(chunks))
        except (IngestError, OSError, UnicodeError) as exc:
            report.errors[source] = str(exc)
            log.warning("failed %s: %s", source, exc)
    if prune and base.is_dir():
        prefix = str(base).rstrip("/") + "/"
        for stored, label in store.document_paths():
            if stored.startswith(prefix) and stored not in seen:
                store.delete_document(stored)
                report.removed.append(label)
    return report
