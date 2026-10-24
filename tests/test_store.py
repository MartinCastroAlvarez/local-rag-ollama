from __future__ import annotations

import stat
from pathlib import Path

import pytest

from local_rag.chunking import Chunk
from local_rag.embeddings import HashingEmbedder
from local_rag.store import Store
from local_rag.store import StoreError
from local_rag.store import fts_query
from local_rag.store import pack
from local_rag.store import unpack


def _chunks(*texts: str) -> list[Chunk]:
    return [Chunk("a.md", i, "A", ("H",), None, t) for i, t in enumerate(texts)]


def _add(store: Store, path: str, *texts: str) -> None:
    emb = HashingEmbedder(64)
    chunks = _chunks(*texts)
    vectors = emb.embed_documents([c.indexed_text for c in chunks])
    store.bind_embedder(emb.name, 64)
    store.add_document(
        path=path,
        source=Path(path).name,
        kind="markdown",
        title="A",
        sha256="h",
        chunks=chunks,
        embeddings=vectors,
    )


@pytest.mark.parametrize("use_vec", [True, False])
def test_bm25_and_vector_search(tmp_path: Path, use_vec: bool) -> None:
    with Store(tmp_path / "i.sqlite3", use_sqlite_vec=use_vec) as store:
        assert store.vec is use_vec
        _add(store, "/a.md", "red apples grow on trees", "blue whales swim in oceans")
        bm = store.search_bm25("whales", 5)
        assert len(bm) == 1
        hit = store.get_chunks([bm[0][0]])[bm[0][0]]
        assert "whales" in hit.text and hit.breadcrumb == "A > H"
        vec = store.search_vector(HashingEmbedder(64).embed_query("apples trees"), 2)
        assert store.get_chunks([vec[0][0]])[vec[0][0]].text.startswith("red apples")
        assert vec[0][1] > vec[1][1]
        assert store.stats()["vector_index"] == ("sqlite-vec" if use_vec else "python-scan")


def test_both_vector_paths_agree(tmp_path: Path) -> None:
    texts = [f"document {i} about topic {i % 3} and colour {i % 5}" for i in range(20)]
    query = HashingEmbedder(64).embed_query("topic 1 colour 3")
    results = []
    for use_vec in (True, False):
        with Store(tmp_path / f"{use_vec}.sqlite3", use_sqlite_vec=use_vec) as store:
            _add(store, "/a.md", *texts)
            results.append([(c, round(s, 4)) for c, s in store.search_vector(query, 5)])
    assert results[0] == results[1]


def test_fts_query_neutralises_operators() -> None:
    assert fts_query('NEAR(") OR * "x" AND title:secret') == '"near" OR "x" OR "title" OR "secret"'
    assert fts_query("the and of") == '"the" OR "and" OR "of"'  # all stopwords: keep them
    assert fts_query("!!!") == ""


def test_hostile_query_strings_do_not_break_fts(tmp_path: Path) -> None:
    with Store(tmp_path / "i.sqlite3") as store:
        _add(store, "/a.md", "plain text")
        for q in ['"', "a OR", "NOT", "*", "col:val", "'; DROP TABLE chunks; --", ""]:
            store.search_bm25(q, 5)
        assert store.stats()["chunks"] == 1


def test_embedder_mismatch_is_refused(tmp_path: Path) -> None:
    with Store(tmp_path / "i.sqlite3") as store:
        with pytest.raises(StoreError, match="empty"):
            store.check_embedder("hashing:64")
        _add(store, "/a.md", "x")
        store.check_embedder("hashing:64")
        with pytest.raises(StoreError, match="built with"):
            store.check_embedder("ollama:nomic-embed-text")
        with pytest.raises(StoreError, match="Re-ingest"):
            store.bind_embedder("hashing:128", 128)


def test_replace_delete_and_clear(tmp_path: Path) -> None:
    with Store(tmp_path / "i.sqlite3") as store:
        _add(store, "/a.md", "first version")
        _add(store, "/a.md", "second version", "another chunk")
        assert store.stats()["chunks"] == 2
        assert store.search_bm25("first", 5) == []
        _add(store, "/b.md", "bee")
        store.delete_document("/a.md")
        store.delete_document("/missing.md")
        assert [d["source"] for d in store.documents()] == ["b.md"]
        assert store.search_vector(HashingEmbedder(64).embed_query("second"), 10)[0][0] in {
            c.id for c in store.all_chunks()
        }
        store.clear()
        assert store.stats()["chunks"] == 0 and store.meta("embedder") is None
        assert store.search_vector([0.0] * 64, 3) == []


def test_index_file_is_private(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "i.sqlite3"
    Store(path).close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_pack_roundtrip_and_length_check(tmp_path: Path) -> None:
    assert unpack(pack([1.0, -2.5])) == [1.0, -2.5]
    with Store(tmp_path / "i.sqlite3") as store, pytest.raises(ValueError):
        store.add_document(
            path="/x",
            source="x",
            kind="text",
            title="t",
            sha256="h",
            chunks=_chunks("a"),
            embeddings=[],
        )
    with Store(":memory:") as memory:
        assert memory.stats()["documents"] == 0
