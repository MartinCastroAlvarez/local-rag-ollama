"""Retrieval modes: BM25, vector, hybrid (reciprocal rank fusion) and hybrid + rerank."""

from __future__ import annotations

import time
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol

from local_rag.config import MODES
from local_rag.config import Settings
from local_rag.embeddings import Embedder
from local_rag.embeddings import _disable_hub_telemetry
from local_rag.embeddings import cache_dir
from local_rag.store import Store
from local_rag.store import StoredChunk


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class FastEmbedReranker:
    """A local ONNX cross-encoder (default ``Xenova/ms-marco-MiniLM-L-6-v2``, ~80 MB)."""

    def __init__(self, model: str, *, threads: int = 0) -> None:
        _disable_hub_telemetry()
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError(
                "the reranker needs fastembed: pip install 'local-rag-ollama[fastembed]'"
            ) from exc
        self.name = f"fastembed:{model}"
        self._model: Any = TextCrossEncoder(
            model_name=model, cache_dir=str(cache_dir()), threads=threads or None
        )

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, list(texts), batch_size=8)]


def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: ``score(d) = sum(1 / (k + rank_i(d)))``, ranks from 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass(frozen=True)
class Hit:
    chunk: StoredChunk
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    similarity: float | None = None
    rerank_score: float | None = None


@dataclass
class Retrieval:
    query: str
    mode: str
    hits: list[Hit]
    timings: dict[str, float] = field(default_factory=dict)
    relevant: bool = True
    """False when nothing retrieved looks related (no BM25 match and low similarity)."""


class Retriever:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        settings: Settings,
        *,
        reranker: Reranker | None = None,
        reranker_factory: Callable[[], Reranker] | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self._reranker = reranker
        self._reranker_factory = reranker_factory or (
            lambda: FastEmbedReranker(settings.rerank_model, threads=settings.threads)
        )

    def load_reranker(self) -> Reranker:
        """The cross-encoder, created on first use (it may download a model once)."""
        if self._reranker is None:
            self._reranker = self._reranker_factory()
        return self._reranker

    def retrieve(
        self, query: str, *, mode: str | None = None, top_k: int | None = None
    ) -> Retrieval:
        mode = mode or self.settings.mode
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        top_k = top_k or self.settings.top_k
        pool = max(self.settings.candidates, top_k)
        timings: dict[str, float] = {}
        bm25: list[tuple[int, float]] = []
        vector: list[tuple[int, float]] = []

        if mode != "vector":
            t0 = time.perf_counter()
            bm25 = self.store.search_bm25(query, pool)
            timings["bm25"] = time.perf_counter() - t0
        if mode != "bm25":
            self.store.check_embedder(self.embedder.name)
            t0 = time.perf_counter()
            qvec = self.embedder.embed_query(query)
            timings["embed_query"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            vector = self.store.search_vector(qvec, pool)
            timings["vector"] = time.perf_counter() - t0

        bm25_rank = {cid: i for i, (cid, _) in enumerate(bm25, start=1)}
        vec_rank = {cid: i for i, (cid, _) in enumerate(vector, start=1)}
        sims = dict(vector)

        if mode == "bm25":
            ranked = bm25
        elif mode == "vector":
            ranked = vector
        else:
            t0 = time.perf_counter()
            ranked = rrf([[c for c, _ in bm25], [c for c, _ in vector]], k=self.settings.rrf_k)
            timings["fuse"] = time.perf_counter() - t0

        keep = pool if mode == "hybrid+rerank" else top_k
        chunks = self.store.get_chunks(cid for cid, _ in ranked[:keep])
        hits = [
            Hit(
                chunk=chunks[cid],
                score=score,
                bm25_rank=bm25_rank.get(cid),
                vector_rank=vec_rank.get(cid),
                similarity=sims.get(cid),
            )
            for cid, score in ranked[:keep]
            if cid in chunks
        ]

        if mode == "hybrid+rerank" and hits:
            t0 = time.perf_counter()
            texts = [f"{h.chunk.breadcrumb}\n\n{h.chunk.text}" for h in hits]
            scores = self.load_reranker().score(query, texts)
            timings["rerank"] = time.perf_counter() - t0
            hits = sorted(
                (
                    Hit(h.chunk, s, h.bm25_rank, h.vector_rank, h.similarity, s)
                    for h, s in zip(hits, scores, strict=True)
                ),
                key=lambda h: -h.score,
            )[:top_k]

        # Relevance gate: refuse up front only when no query term occurs anywhere in the
        # index *and* the nearest vector is far away. Everything else goes to the model,
        # which is told to refuse when the sources do not answer the question.
        keyword_match = bool(bm25) if mode != "vector" else bool(self.store.search_bm25(query, 1))
        best_sim = max(sims.values(), default=None)
        relevant = bool(hits) and (
            keyword_match or best_sim is None or best_sim >= self.settings.min_similarity
        )
        return Retrieval(query=query, mode=mode, hits=hits, timings=timings, relevant=relevant)
