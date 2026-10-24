"""Wire settings into a ready-to-use store, embedder, retriever and RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from local_rag.config import Settings
from local_rag.embeddings import Embedder
from local_rag.embeddings import make_embedder
from local_rag.ollama import OllamaClient
from local_rag.rag import RAG
from local_rag.rag import ChatModel
from local_rag.rag import OllamaChat
from local_rag.retrieval import Reranker
from local_rag.retrieval import Retriever
from local_rag.store import Store


@dataclass
class Runtime:
    settings: Settings
    client: OllamaClient
    store: Store
    embedder: Embedder
    retriever: Retriever
    rag: RAG

    def close(self) -> None:
        self.store.close()
        self.client.close()

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def settings_for_index(settings: Settings, stored: str | None) -> Settings:
    kind, _, model = (stored or "").partition(":")
    if kind == "hashing":
        return settings.with_(embedder="hashing", hashing_dim=int(model))
    if kind == "fastembed":
        return settings.with_(embedder="fastembed", fastembed_model=model)
    if kind == "ollama":
        return settings.with_(embedder="ollama", embed_model=model)
    return settings


def open_runtime(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    llm: ChatModel | None = None,
    reranker: Reranker | None = None,
    client: OllamaClient | None = None,
    follow_index: bool = True,
) -> Runtime:
    """Build the pipeline. With ``follow_index`` the query embedder is the one that built
    the index (recorded in its meta table), whatever the current settings say: querying
    with a different model would return noise."""
    client = client or OllamaClient(settings.ollama_url, timeout_s=settings.timeout_s)
    store = Store(settings.db_path)
    if embedder is None and follow_index:
        settings = settings_for_index(settings, store.meta("embedder"))
    if embedder is None:
        embedder = make_embedder(settings, client)
    llm = llm or OllamaChat(client, settings.chat_model, temperature=settings.temperature)
    retriever = Retriever(store, embedder, settings, reranker=reranker)
    return Runtime(settings, client, store, embedder, retriever, RAG(retriever, llm, settings))
