from __future__ import annotations

from pathlib import Path

import pytest

from local_rag.chunking import chunk_document
from local_rag.config import Settings
from local_rag.embeddings import HashingEmbedder
from local_rag.evaluation import bundled_corpus
from local_rag.evaluation import evaluate_generation
from local_rag.evaluation import is_relevant
from local_rag.evaluation import judge_faithfulness
from local_rag.evaluation import load_qa
from local_rag.evaluation import markdown_table
from local_rag.evaluation import run_eval
from local_rag.rag import RAG
from local_rag.retrieval import Retriever
from local_rag.store import StoredChunk
from tests.conftest import FakeJudge
from tests.conftest import FakeLLM
from tests.conftest import FakeReranker


def test_bundled_qa_set_shape() -> None:
    items = load_qa()
    assert len(items) >= 40
    assert len({i.id for i in items}) == len(items)
    sources = {p.name for p in bundled_corpus().iterdir()}
    assert {
        "us-constitution.md",
        "art-of-war.txt",
        "elements-of-style.html",
        "civil-disobedience.pdf",
    } <= sources
    for item in items:
        if item.answerable:
            assert item.evidence and item.source in sources
        else:
            assert not item.evidence


def test_every_evidence_span_survives_default_chunking() -> None:
    result = run_eval(Settings(embedder="hashing"), HashingEmbedder(256), modes=["bm25"])
    assert result["unmatched_evidence"] == []
    assert result["corpus"]["documents"] == 4 and not result["corpus"]["errors"]


def test_relevance_requires_the_right_source() -> None:
    item = load_qa()[0]
    good = StoredChunk(1, item.source or "", "t", (), None, f"xx {item.evidence[0].upper()} yy")
    wrong = StoredChunk(2, "other.md", "t", (), None, item.evidence[0])
    assert is_relevant(good, item) and not is_relevant(wrong, item)


def test_run_eval_all_modes_with_fakes(tmp_path: Path) -> None:
    result = run_eval(
        Settings(embedder="hashing"),
        HashingEmbedder(256),
        llm=FakeLLM(),
        judge=FakeJudge(),
        reranker_factory=FakeReranker,
    )
    retrieval = result["retrieval"]
    assert set(retrieval) == {"bm25", "vector", "hybrid", "hybrid+rerank"}
    for metrics in retrieval.values():
        assert 0 <= metrics["recall@1"] <= metrics["recall@5"] <= metrics["recall@10"] <= 1
        assert 0 < metrics["mrr@10"] <= 1
        assert metrics["latency_ms"]["total"] >= 0
    assert retrieval["bm25"]["recall@10"] > 0.9
    gen = result["generation"]
    assert gen["faithfulness"] == 1.0 and gen["citation_rate"] == 1.0
    table = markdown_table(result)
    assert "| hybrid+rerank |" in table and "Faithfulness" in table


def test_missing_reranker_is_reported_not_fatal() -> None:
    def broken() -> FakeReranker:
        raise RuntimeError("fastembed is not installed")

    result = run_eval(
        Settings(embedder="hashing"),
        HashingEmbedder(64),
        modes=["hybrid+rerank"],
        reranker_factory=broken,
    )
    assert result["retrieval"]["hybrid+rerank"] == {"skipped": "fastembed is not installed"}
    assert "skipped" in markdown_table(result)


@pytest.mark.parametrize(
    ("reply", "score"),
    [
        ('{"verdict": "supported"}', 1.0),
        ('Sure! {"verdict": "partial"}', 0.5),
        ("unsupported, clearly", 0.0),
        ("{not json", None),
        ('{"verdict": "maybe"}', None),
    ],
)
def test_judge_parsing(reply: str, score: float | None) -> None:
    chunk = StoredChunk(1, "a", "t", (), None, "text")
    assert judge_faithfulness(FakeJudge(reply), [chunk], "answer") == score


def test_generation_metrics(retriever: Retriever, settings: Settings) -> None:
    from local_rag.evaluation import QAItem

    items = [
        QAItem(
            "a",
            "How much sun for tomatoes?",
            "",
            ("six hours of direct sun",),
            "guides/garden.md",
            True,
        ),
        QAItem("b", "What about basil?", "", ("pinched above a leaf",), "guides/garden.md", True),
        QAItem("u", "Who won the 2018 world cup?", "", (), None, False),
    ]
    llm = FakeLLM(["Six hours [1].", settings.refusal, settings.refusal])
    judge = FakeJudge('{"verdict": "partial"}')
    gen = evaluate_generation(RAG(retriever, llm, settings), items, mode="hybrid", judge=judge)
    assert gen["answer_rate"] == 0.5
    assert gen["refusal_rate_unanswerable"] == 1.0
    assert gen["citation_rate"] == 1.0 and gen["citation_precision"] == 1.0
    assert gen["faithfulness"] == 0.5 and judge.calls == 1
    no_judge = evaluate_generation(
        RAG(retriever, FakeLLM(), settings), items[:1], mode="bm25", judge=None
    )
    assert no_judge["faithfulness"] is None


def test_custom_qa_file(tmp_path: Path, corpus: Path) -> None:
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        '{"id": "1", "question": "Saturn rings?", "evidence": ["Saturn has bright rings"], "source": "astronomy.pdf"}\n\n',
        encoding="utf-8",
    )
    result = run_eval(
        Settings(embedder="hashing"), HashingEmbedder(64), modes=["bm25"], corpus=corpus, qa_path=qa
    )
    assert result["retrieval"]["bm25"]["recall@1"] == 1.0


def test_chunk_sizes_on_bundled_corpus() -> None:
    from local_rag.loaders import discover
    from local_rag.loaders import load

    for path, source in discover(bundled_corpus(), max_bytes=10**8):
        chunks = chunk_document(
            load(path, source, max_bytes=10**8), chunk_chars=1200, overlap_chars=200
        )
        assert chunks and all(len(c.text) <= 1200 for c in chunks)
