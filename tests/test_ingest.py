from __future__ import annotations

from pathlib import Path

from local_rag.config import Settings
from local_rag.embeddings import HashingEmbedder
from local_rag.ingest import ingest
from local_rag.store import Store


def test_ingest_is_incremental(corpus: Path, store: Store, settings: Settings) -> None:
    emb = HashingEmbedder(settings.hashing_dim)
    first = ingest(corpus, settings=settings, store=store, embedder=emb)
    assert sorted(first.added) == ["astronomy.pdf", "bikes.html", "bread.txt", "guides/garden.md"]
    assert first.chunks == store.stats()["chunks"] > 0 and not first.errors
    again = ingest(corpus, settings=settings, store=store, embedder=emb)
    assert again.added == [] and len(again.unchanged) == 4
    (corpus / "bread.txt").write_text("RYE\n\nRye bread is dense.\n", encoding="utf-8")
    third = ingest(corpus, settings=settings, store=store, embedder=emb)
    assert third.added == ["bread.txt"]
    assert store.search_bm25("sourdough", 5) == []
    assert "1 indexed" in third.summary()


def test_changing_chunk_settings_reindexes(corpus: Path, store: Store, settings: Settings) -> None:
    emb = HashingEmbedder(settings.hashing_dim)
    ingest(corpus, settings=settings, store=store, embedder=emb)
    bigger = settings.with_(chunk_chars=2000)
    assert len(ingest(corpus, settings=bigger, store=store, embedder=emb).added) == 4


def test_prune_forgets_deleted_files(corpus: Path, store: Store, settings: Settings) -> None:
    emb = HashingEmbedder(settings.hashing_dim)
    ingest(corpus, settings=settings, store=store, embedder=emb)
    (corpus / "bikes.html").unlink()
    kept = ingest(corpus, settings=settings, store=store, embedder=emb)
    assert kept.removed == [] and store.stats()["documents"] == 4
    pruned = ingest(corpus, settings=settings, store=store, embedder=emb, prune=True)
    assert pruned.removed == ["bikes.html"] and store.stats()["documents"] == 3


def test_unreadable_and_empty_files_are_reported(
    corpus: Path, store: Store, settings: Settings
) -> None:
    (corpus / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    (corpus / "empty.md").write_text("   \n")
    report = ingest(corpus, settings=settings, store=store, embedder=HashingEmbedder(32))
    assert set(report.errors) == {"broken.pdf", "empty.md"}
    assert "no extractable text" in report.errors["empty.md"]
    assert len(report.added) == 4
