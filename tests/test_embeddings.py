from __future__ import annotations

import math

from local_rag.embeddings import HashingEmbedder
from local_rag.embeddings import normalise
from local_rag.embeddings import stem
from local_rag.embeddings import terms


def test_hashing_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(128)
    a = emb.embed_query("The Senate tries impeachments")
    assert a == emb.embed_documents(["The Senate tries impeachments"])[0]
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)
    assert emb.name == "hashing:128" and len(a) == 128


def _dot(u: list[float], v: list[float]) -> float:
    return sum(x * y for x, y in zip(u, v, strict=True))


def test_similar_texts_score_higher() -> None:
    emb = HashingEmbedder(512)
    q = emb.embed_query("impeachment trial senate")
    near = emb.embed_query("the senate holds the impeachment trial")
    far = emb.embed_query("bake sourdough bread in a dutch oven")
    assert _dot(q, near) > _dot(q, far)


def test_stemming_and_stopwords() -> None:
    assert stem("parties") == "party" and stem("carried") == "carry" and stem("running") == "runn"
    assert stem("is") == "is"
    assert terms("The cats are running") == ["cat", "runn"]


def test_normalise_zero_vector() -> None:
    assert normalise([0.0, 0.0]) == [0.0, 0.0]
    assert HashingEmbedder(16).embed_query("the and of") == [0.0] * 16
