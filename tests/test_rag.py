from __future__ import annotations

import pytest

from local_rag.rag import RAG
from local_rag.retrieval import Retriever
from tests.conftest import FakeLLM


def test_stream_event_order_and_citations(rag: RAG, llm: FakeLLM) -> None:
    llm.replies = ["Chains need oil every 300 km [1]. Also see [9]."]
    events = list(rag.stream("How often do I lubricate the chain?"))
    assert events[0]["type"] == "sources" and events[0]["sources"][0]["n"] == 1
    assert {e["type"] for e in events[1:-1]} == {"token"}
    done = events[-1]
    assert done["type"] == "done"
    assert done["citations"] == [1]
    assert "[9]" not in done["text"] and done["text"].endswith("Also see.")
    assert not done["refused"]
    assert {"retrieve", "generate", "first_token"} <= set(done["timings"])


def test_ask_returns_citation_objects(rag: RAG, llm: FakeLLM) -> None:
    llm.replies = ["Six hours of sun [1][2]."]
    answer = rag.ask("How much sun do tomatoes need?")
    assert [c.number for c in answer.citations] == [1, 2]
    assert answer.citations[0].chunk.id == answer.retrieval.hits[0].chunk.id
    assert answer.sources[0]["source"] == "guides/garden.md"
    assert not answer.refused


def test_model_refusal_is_detected(rag: RAG, llm: FakeLLM) -> None:
    llm.replies = ["I don't know based on the provided documents."]
    answer = rag.ask("What colour is the sky on Mars?", mode="bm25")
    assert answer.refused and answer.citations == []


def test_irrelevant_retrieval_refuses_without_calling_the_model(
    retriever: Retriever, llm: FakeLLM, settings
) -> None:
    rag = RAG(retriever, llm, settings)
    answer = rag.ask("zzzz qqqq", mode="vector")
    assert answer.refused and llm.calls == []
    assert answer.text == settings.refusal


def test_empty_answer_counts_as_refusal(rag: RAG, llm: FakeLLM) -> None:
    llm.replies = ["   "]
    answer = rag.ask("tomatoes")
    assert answer.refused and answer.text == rag.settings.refusal


def test_history_is_sent_and_used_for_retrieval(rag: RAG, llm: FakeLLM) -> None:
    llm.replies = ["ok [1]"]
    rag.ask("and basil?", history=[("How do I grow tomatoes?", "Sun [1].")])
    messages = llm.calls[0]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "How do I grow tomatoes?"


def test_empty_question_rejected(rag: RAG) -> None:
    with pytest.raises(ValueError):
        rag.ask("   ")
