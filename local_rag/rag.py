"""The question-answering pipeline: retrieve -> prompt -> stream -> validate citations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol

from local_rag.config import Settings
from local_rag.ollama import OllamaClient
from local_rag.prompting import Citation
from local_rag.prompting import build_messages
from local_rag.prompting import clean_citations
from local_rag.prompting import is_refusal
from local_rag.retrieval import Retrieval
from local_rag.retrieval import Retriever


class ChatModel(Protocol):
    name: str

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]: ...


class OllamaChat:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        *,
        temperature: float = 0.0,
        json_format: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.json_format = json_format
        self.name = f"ollama:{model}"

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        return self.client.chat(
            self.model, messages, temperature=self.temperature, json_format=self.json_format
        )


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    refused: bool
    retrieval: Retrieval
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def sources(self) -> list[dict[str, Any]]:
        return [source_payload(n, h.chunk) for n, h in enumerate(self.retrieval.hits, start=1)]


def source_payload(n: int, chunk: Any) -> dict[str, Any]:
    return {
        "n": n,
        "chunk_id": chunk.id,
        "source": chunk.source,
        "location": chunk.breadcrumb,
        "page": chunk.page,
        "preview": chunk.text[:240],
    }


class RAG:
    def __init__(self, retriever: Retriever, llm: ChatModel, settings: Settings) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    def stream(
        self,
        question: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        history: Sequence[tuple[str, str]] = (),
    ) -> Iterator[dict[str, Any]]:
        """Yield a ``sources`` event, then ``token`` events, then one ``done`` event."""
        for event in self._run(question, mode=mode, top_k=top_k, history=history):
            if isinstance(event, dict):
                yield event

    def ask(
        self,
        question: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        history: Sequence[tuple[str, str]] = (),
    ) -> Answer:
        retrieval: Retrieval | None = None
        done: dict[str, Any] = {}
        for event in self._run(question, mode=mode, top_k=top_k, history=history):
            if isinstance(event, Retrieval):
                retrieval = event
            elif event["type"] == "done":
                done = event
        if retrieval is None or not done:  # pragma: no cover - _run always yields both
            raise RuntimeError("pipeline produced no answer")
        return Answer(
            question=question,
            text=str(done["text"]),
            citations=[Citation(n, retrieval.hits[n - 1].chunk) for n in done["citations"]],
            refused=bool(done["refused"]),
            retrieval=retrieval,
            timings=dict(done["timings"]),
        )

    def _run(
        self,
        question: str,
        *,
        mode: str | None,
        top_k: int | None,
        history: Sequence[tuple[str, str]],
    ) -> Iterator[Retrieval | dict[str, Any]]:
        question = " ".join(question.split())[:2000]
        if not question:
            raise ValueError("empty question")
        t0 = time.perf_counter()
        # A follow-up ("and the Senate?") retrieves better with the previous question attached.
        query = f"{history[-1][0]} {question}" if history else question
        retrieval = self.retriever.retrieve(query, mode=mode, top_k=top_k)
        yield retrieval
        timings = dict(retrieval.timings)
        timings["retrieve"] = time.perf_counter() - t0
        hits = retrieval.hits
        yield {
            "type": "sources",
            "mode": retrieval.mode,
            "sources": [source_payload(n, h.chunk) for n, h in enumerate(hits, start=1)],
        }
        if not hits or not retrieval.relevant:
            text = self.settings.refusal
            yield {"type": "token", "text": text}
            timings["generate"] = 0.0
            yield {
                "type": "done",
                "text": text,
                "citations": [],
                "refused": True,
                "timings": timings,
            }
            return

        messages = build_messages(
            question, [h.chunk for h in hits], refusal=self.settings.refusal, history=history
        )
        t0 = time.perf_counter()
        first: float | None = None
        parts: list[str] = []
        for token in self.llm.stream(messages):
            if first is None:
                first = time.perf_counter() - t0
            parts.append(token)
            yield {"type": "token", "text": token}
        timings["generate"] = time.perf_counter() - t0
        timings["first_token"] = first if first is not None else timings["generate"]
        text, numbers = clean_citations("".join(parts).strip(), len(hits))
        refused = not text or is_refusal(text, self.settings.refusal)
        yield {
            "type": "done",
            "text": text or self.settings.refusal,
            "citations": [] if refused else numbers,
            "refused": refused,
            "timings": timings,
        }
