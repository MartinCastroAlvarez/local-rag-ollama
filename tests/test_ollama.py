from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from local_rag.config import Settings
from local_rag.embeddings import OllamaEmbedder
from local_rag.embeddings import make_embedder
from local_rag.ollama import OllamaClient
from local_rag.ollama import OllamaError
from local_rag.rag import OllamaChat


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaClient:
    return OllamaClient("http://127.0.0.1:11434", transport=httpx.MockTransport(handler))


def test_embed_batches_prefixes_and_normalises() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0] for _ in body["input"]]})

    emb = OllamaEmbedder(client_for(handler), "nomic-embed-text:latest", batch_size=2)
    vectors = emb.embed_documents(["a", "b", "c"])
    assert vectors == [[0.6, 0.8]] * 3
    assert [len(b["input"]) for b in seen] == [2, 1]
    assert seen[0]["input"][0] == "search_document: a" and seen[0]["truncate"] is True
    assert emb.embed_query("q") == [0.6, 0.8]
    assert seen[-1]["input"] == ["search_query: q"]
    assert emb.name == "ollama:nomic-embed-text:latest"


def test_models_without_prefixes() -> None:
    emb = OllamaEmbedder(client_for(lambda r: httpx.Response(200)), "bge-m3")
    assert (emb.doc_prefix, emb.query_prefix) == ("", "")


def test_embed_count_mismatch_and_http_errors() -> None:
    bad = client_for(lambda r: httpx.Response(200, json={"embeddings": []}))
    with pytest.raises(OllamaError, match="wrong number"):
        bad.embed("m", ["x"])
    missing = client_for(lambda r: httpx.Response(404, json={"error": "model 'm' not found"}))
    with pytest.raises(OllamaError, match="not found"):
        missing.embed("m", ["x"])
    weird = client_for(lambda r: httpx.Response(200, json=[1, 2]))
    with pytest.raises(OllamaError, match="unexpected"):
        weird.embed("m", ["x"])
    text_err = client_for(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(OllamaError, match="boom"):
        text_err.embed("m", ["x"])


def test_connection_error_has_a_hint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(OllamaError, match="ollama pull m"):
        client_for(handler).embed("m", ["x"])
    with pytest.raises(OllamaError, match="ollama serve"):
        client_for(handler).models()
    with pytest.raises(OllamaError, match="ollama pull m"):
        list(client_for(handler).chat("m", []))


def test_chat_streams_content_and_sends_options() -> None:
    sent: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        lines = [
            {"message": {"role": "assistant", "content": "Hel"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": False},
            {"message": {"role": "assistant", "content": "lo"}, "done": False},
            {"done": True},
            {"message": {"content": "ignored"}},
        ]
        body = (
            "\n".join(json.dumps(x) for x in lines[:3])
            + "\n\n"
            + "\n".join(json.dumps(x) for x in lines[3:])
        )
        return httpx.Response(200, content=body.encode())

    chat = OllamaChat(client_for(handler), "qwen3:4b", temperature=0.2, json_format=True)
    assert "".join(chat.stream([{"role": "user", "content": "hi"}])) == "Hello"
    assert sent["stream"] is True and sent["think"] is False
    assert sent["options"] == {"temperature": 0.2} and sent["format"] == "json"
    assert chat.name == "ollama:qwen3:4b"


def test_chat_errors() -> None:
    err_status = client_for(lambda r: httpx.Response(404, json={"error": "no model"}))
    with pytest.raises(OllamaError, match="no model"):
        list(err_status.chat("m", []))
    err_event = client_for(lambda r: httpx.Response(200, content=b'{"error": "oom"}\n'))
    with pytest.raises(OllamaError, match="oom"):
        list(err_event.chat("m", []))


def test_redirects_are_never_followed() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://evil.example/collect"})

    with pytest.raises(OllamaError, match="HTTP 302"):
        client_for(handler).embed("m", ["secret document text"])
    assert calls == ["http://127.0.0.1:11434/api/embed"]


def test_models_listing() -> None:
    ok = client_for(lambda r: httpx.Response(200, json={"models": [{"name": "qwen3:4b"}]}))
    assert ok.models() == ["qwen3:4b"]
    bad = client_for(lambda r: httpx.Response(500))
    with pytest.raises(OllamaError):
        bad.models()
    ok.close()


def test_make_embedder_picks_backend() -> None:
    assert make_embedder(Settings(embedder="hashing", hashing_dim=32)).name == "hashing:32"
    emb = make_embedder(Settings(embedder="ollama", embed_model="bge-m3"))
    assert emb.name == "ollama:bge-m3"
