"""Shared fakes. Nothing here touches the network or a real model."""

from __future__ import annotations

import re
from collections.abc import Iterator
from collections.abc import Sequence
from pathlib import Path

import pytest

from local_rag.config import Settings
from local_rag.embeddings import HashingEmbedder
from local_rag.ingest import ingest
from local_rag.rag import RAG
from local_rag.retrieval import Retriever
from local_rag.store import Store


class FakeLLM:
    """Answers from a script, or by citing source 1. Records every prompt it receives."""

    name = "fake:llm"

    def __init__(self, replies: Sequence[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[list[dict[str, str]]] = []

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        self.calls.append(messages)
        reply = self.replies.pop(0) if self.replies else "The answer is in the documents [1]."
        # Stream in small pieces, like a real model.
        yield from re.findall(r"\S+\s*|\s+", reply)


class FakeJudge:
    name = "fake:judge"

    def __init__(self, verdict: str = '{"verdict": "supported"}') -> None:
        self.verdict = verdict
        self.calls = 0

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        self.calls += 1
        yield self.verdict


class FakeReranker:
    """Scores by how many query words appear in the text."""

    name = "fake:reranker"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        words = set(re.findall(r"\w+", query.lower()))
        return [float(len(words & set(re.findall(r"\w+", t.lower())))) for t in texts]


def make_pdf(pages: Sequence[str], title: str | None = None) -> bytes:
    """A minimal, valid PDF with one text line per page (Helvetica, WinAnsi)."""
    objects: list[bytes] = []
    n_pages = len(pages)
    page_ids = [3 + 2 * i for i in range(n_pages)]
    font_id = 3 + 2 * n_pages
    info_id = font_id + 1
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{p} 0 R" for p in page_ids).encode()
    objects.append(
        b"<< /Type /Pages /Kids [" + kids + b"] /Count " + str(n_pages).encode() + b" >>"
    )
    for i, text in enumerate(pages):
        content_id = page_ids[i] + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("latin-1")
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    )
    objects.append(f"<< /Title ({title or ''}) >>".encode())
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {info_id} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in list(__import__("os").environ):
        if key.startswith("LOCAL_RAG_") or key == "OLLAMA_HOST":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "index.sqlite3", embedder="hashing", hashing_dim=256)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "guides").mkdir(parents=True)
    (root / "guides" / "garden.md").write_text(
        "# Gardening\n\n## Tomatoes\n\nTomatoes need six hours of direct sun every day.\n\n"
        "## Basil\n\nBasil should be pinched above a leaf pair to keep it bushy.\n",
        encoding="utf-8",
    )
    (root / "bread.txt").write_text(
        "BREAD BASICS\n\nSourdough starter is fed with equal weights of flour and water.\n\n"
        "BAKING\n\nBake the loaf in a preheated Dutch oven at 250 degrees for twenty minutes.\n",
        encoding="utf-8",
    )
    (root / "bikes.html").write_text(
        "<html><head><title>Bike care</title><script>alert('x')</script></head><body>"
        "<h1>Bike care</h1><h2>Chains</h2><p>Lubricate the chain every 300 kilometres.</p>"
        "<h2>Tyres</h2><p>Road tyres are pumped to about 6 bar.</p></body></html>",
        encoding="utf-8",
    )
    (root / "astronomy.pdf").write_bytes(
        make_pdf(
            ["Jupiter has the Great Red Spot, a giant storm.", "Saturn has bright rings."],
            title="Planets",
        )
    )
    return root


@pytest.fixture
def store(settings: Settings) -> Iterator[Store]:
    s = Store(settings.db_path)
    yield s
    s.close()


@pytest.fixture
def indexed(store: Store, corpus: Path, settings: Settings) -> Store:
    ingest(corpus, settings=settings, store=store, embedder=HashingEmbedder(settings.hashing_dim))
    return store


@pytest.fixture
def retriever(indexed: Store, settings: Settings) -> Retriever:
    return Retriever(
        indexed, HashingEmbedder(settings.hashing_dim), settings, reranker=FakeReranker()
    )


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def rag(retriever: Retriever, llm: FakeLLM, settings: Settings) -> RAG:
    return RAG(retriever, llm, settings)
