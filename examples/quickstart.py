"""Index the bundled corpus and query it from Python, no ollama needed.

    python -m examples.quickstart

Uses the dependency-free hashing embedder and a temporary index. With ollama running
(and `ollama pull qwen3:4b`), it also prints a cited answer.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from local_rag.config import Settings
from local_rag.embeddings import HashingEmbedder
from local_rag.evaluation import bundled_corpus
from local_rag.ingest import ingest
from local_rag.ollama import OllamaError
from local_rag.runtime import open_runtime

QUESTIONS = [
    "Who has the sole power to try impeachments?",
    "What does Sun Tzu say all warfare is based on?",
    "Why was Thoreau put in jail, and for how long?",
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(db_path=Path(tmp) / "index.sqlite3", embedder="hashing")
        embedder = HashingEmbedder(settings.hashing_dim)
        with open_runtime(settings, embedder=embedder) as rt:
            report = ingest(bundled_corpus(), settings=settings, store=rt.store, embedder=embedder)
            print(f"ingest: {report.summary()} [{rt.store.stats()['vector_index']}]")
            for question in QUESTIONS:
                hits = rt.retriever.retrieve(question, mode="hybrid", top_k=3).hits
                print(f"\nQ: {question}")
                for n, hit in enumerate(hits, start=1):
                    print(f"  [{n}] {hit.chunk.source} - {hit.chunk.breadcrumb}")
            try:
                answer = rt.rag.ask(QUESTIONS[0])
            except OllamaError as exc:
                print(f"\n(no answer generated: {exc})")
            else:
                print(f"\nA: {answer.text}")


if __name__ == "__main__":
    main()
