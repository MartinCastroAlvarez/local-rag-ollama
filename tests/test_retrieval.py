from __future__ import annotations

import pytest

from local_rag.config import MODES
from local_rag.config import Settings
from local_rag.embeddings import HashingEmbedder
from local_rag.retrieval import Retriever
from local_rag.retrieval import rrf
from local_rag.store import Store
from tests.conftest import FakeReranker


def test_rrf_math() -> None:
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    scores = dict(fused)
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[2] == pytest.approx(1 / 62)
    assert [c for c, _ in fused] == [1, 3, 2]
    assert rrf([]) == []


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_finds_the_obvious_chunk(retriever: Retriever, mode: str) -> None:
    result = retriever.retrieve(
        "How often should the bike chain be lubricated?", mode=mode, top_k=3
    )
    assert result.mode == mode
    assert result.hits and "Lubricate the chain" in result.hits[0].chunk.text
    assert result.relevant
    assert "total" not in result.timings
    if mode != "vector":
        assert "bm25" in result.timings
    if mode != "bm25":
        assert {"embed_query", "vector"} <= set(result.timings)


def test_hybrid_records_both_ranks(retriever: Retriever) -> None:
    hit = retriever.retrieve("sourdough starter flour water", mode="hybrid").hits[0]
    assert hit.bm25_rank == 1 and hit.vector_rank == 1 and hit.similarity is not None


def test_rerank_reorders_and_truncates(retriever: Retriever) -> None:
    result = retriever.retrieve("Saturn rings", mode="hybrid+rerank", top_k=2)
    assert len(result.hits) <= 2
    assert result.hits[0].rerank_score is not None
    assert "Saturn" in result.hits[0].chunk.text
    assert "rerank" in result.timings


def test_reranker_is_created_lazily(indexed: Store, settings: Settings) -> None:
    made: list[int] = []

    def factory() -> FakeReranker:
        made.append(1)
        return FakeReranker()

    r = Retriever(
        indexed, HashingEmbedder(settings.hashing_dim), settings, reranker_factory=factory
    )
    r.retrieve("tomatoes", mode="hybrid")
    assert made == []
    r.retrieve("tomatoes", mode="hybrid+rerank")
    r.retrieve("basil", mode="hybrid+rerank")
    assert made == [1]


def test_unrelated_query_in_vector_mode_is_not_relevant(retriever: Retriever) -> None:
    result = retriever.retrieve("quantum chromodynamics lattice", mode="vector")
    assert not result.relevant


def test_unknown_mode(retriever: Retriever) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve("x", mode="magic")


def test_empty_index_bm25_mode_has_no_hits(store: Store, settings: Settings) -> None:
    r = Retriever(store, HashingEmbedder(settings.hashing_dim), settings)
    result = r.retrieve("anything", mode="bm25")
    assert result.hits == [] and not result.relevant
