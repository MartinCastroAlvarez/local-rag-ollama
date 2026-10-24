from __future__ import annotations

from local_rag.config import Settings
from local_rag.prompting import build_messages
from local_rag.prompting import cited_numbers
from local_rag.prompting import clean_citations
from local_rag.prompting import is_refusal
from local_rag.runtime import settings_for_index
from local_rag.store import StoredChunk

REFUSAL = "I don't know based on the provided documents."


def test_cited_numbers_handles_lists() -> None:
    assert cited_numbers("a [1] b [2, 3] c [x] [4,5]") == [1, 2, 3, 4, 5]


def test_clean_citations_keeps_valid_and_drops_invented() -> None:
    text, valid = clean_citations("Yes [1, 7]. No [0] . Maybe [3][2].", 3)
    assert text == "Yes [1]. No. Maybe [3][2]."
    assert valid == [1, 2, 3]


def test_refusal_detection() -> None:
    assert is_refusal(REFUSAL, REFUSAL)
    assert is_refusal("I don't know based on the provided documents", REFUSAL)
    assert is_refusal("I DON'T KNOW based on the provided documents. Sorry.", REFUSAL)
    assert not is_refusal("I don't know based on the provided documents, but [1] says x.", REFUSAL)
    assert not is_refusal("Tomatoes need sun [1].", REFUSAL)
    assert not is_refusal("anything", "")


def test_history_is_capped_and_truncated() -> None:
    chunk = StoredChunk(1, "a.md", "A", (), None, "text")
    history = [(f"q{i}", "a" * 5000) for i in range(5)]
    messages = build_messages("now?", [chunk], refusal=REFUSAL, history=history)
    assert [m["content"] for m in messages if m["role"] == "user"][:3] == ["q2", "q3", "q4"]
    assert all(len(m["content"]) <= 2000 for m in messages if m["role"] == "assistant")


def test_settings_follow_the_index() -> None:
    base = Settings(embedder="ollama")
    assert settings_for_index(base, None) is base
    assert settings_for_index(base, "hashing:64").hashing_dim == 64
    fe = settings_for_index(base, "fastembed:BAAI/bge-small-en-v1.5")
    assert (fe.embedder, fe.fastembed_model) == ("fastembed", "BAAI/bge-small-en-v1.5")
    ol = settings_for_index(Settings(embedder="hashing"), "ollama:bge-m3")
    assert (ol.embedder, ol.embed_model) == ("ollama", "bge-m3")
