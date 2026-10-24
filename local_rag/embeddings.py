"""Embedders: ollama (default), fastembed (ONNX, optional extra) and a hashing fallback.

All three return L2-normalised float vectors, so cosine distance in the index is
``1 - dot``. ``name`` identifies model *and* settings; the index refuses to mix names.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Protocol

from local_rag.config import Settings
from local_rag.ollama import OllamaClient

# Task prefixes some embedding models were trained with (document, query).
PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
    "snowflake-arctic-embed": ("", "Represent this sentence for searching relevant passages: "),
    "snowflake-arctic-embed2": ("", "query: "),
}


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def normalise(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [x / norm for x in vector]


class OllamaEmbedder:
    def __init__(self, client: OllamaClient, model: str, *, batch_size: int = 32) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.name = f"ollama:{model}"
        base = model.split(":", 1)[0].rsplit("/", 1)[-1]
        self.doc_prefix, self.query_prefix = PREFIXES.get(base, ("", ""))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [self.doc_prefix + t for t in texts[i : i + self.batch_size]]
            out.extend(normalise(v) for v in self.client.embed(self.model, batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        return normalise(self.client.embed(self.model, [self.query_prefix + text])[0])


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "local-rag"


def _disable_hub_telemetry() -> None:
    # fastembed downloads ONNX weights from the Hugging Face Hub once; keep the hub quiet.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")


class FastEmbedEmbedder:
    """ONNX embeddings on CPU via fastembed (``pip install 'local-rag-ollama[fastembed]'``)."""

    def __init__(self, model: str, *, threads: int = 0) -> None:
        _disable_hub_telemetry()
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError(
                "fastembed is not installed: pip install 'local-rag-ollama[fastembed]'"
            ) from exc
        self.model = model
        self.name = f"fastembed:{model}"
        self._model: Any = TextEmbedding(
            model_name=model, cache_dir=str(cache_dir()), threads=threads or None
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [normalise(v.tolist()) for v in self._model.embed(list(texts), batch_size=8)]

    def embed_query(self, text: str) -> list[float]:
        return normalise(next(iter(self._model.query_embed(text))).tolist())


_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = """a an and are as at be but by for from has have he her his i if in into is it
its of on or our she so than that the their them then there these they this to was we were
what when which who whom why will with would you your shall not no do does did"""
STOPWORDS = frozenset(_STOPWORDS.split())


def stem(word: str) -> str:
    for suffix in ("ingly", "edly", "ing", "ies", "ied", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix in ("ies", "ied"):
                return word[: -len(suffix)] + "y"
            return word[: -len(suffix)]
    return word


def terms(text: str) -> list[str]:
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


class HashingEmbedder:
    """Signed feature hashing of stemmed unigrams and bigrams.

    No model, no download, fully deterministic: the always-available fallback and the
    embedder the tests use. It is a *lexical* vector (no synonyms), so expect it to track
    BM25 rather than beat it.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _vector(self, text: str) -> list[float]:
        words = terms(text)
        features = Counter(words)
        features.update(f"{a} {b}" for a, b in zip(words, words[1:], strict=False))
        vec = [0.0] * self.dim
        for feature, count in features.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            sign = 1.0 if value & 1 else -1.0
            vec[(value >> 1) % self.dim] += sign * (1.0 + math.log(count))
        return normalise(vec)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_embedder(settings: Settings, client: OllamaClient | None = None) -> Embedder:
    if settings.embedder == "hashing":
        return HashingEmbedder(settings.hashing_dim)
    if settings.embedder == "fastembed":
        return FastEmbedEmbedder(settings.fastembed_model, threads=settings.threads)
    if client is None:
        client = OllamaClient(settings.ollama_url, timeout_s=settings.timeout_s)
    return OllamaEmbedder(client, settings.embed_model)
