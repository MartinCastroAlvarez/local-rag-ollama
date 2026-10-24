from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from local_rag.config import Settings
from local_rag.embeddings import HashingEmbedder
from local_rag.ingest import ingest
from local_rag.rag import RAG
from local_rag.runtime import open_runtime
from local_rag.web import create_app
from tests.conftest import FakeLLM
from tests.conftest import FakeReranker

BASE = "http://127.0.0.1:8765"


@pytest.fixture
def client(corpus: Path, settings: Settings) -> Iterator[TestClient]:
    emb = HashingEmbedder(settings.hashing_dim)
    runtime = open_runtime(settings, embedder=emb, llm=FakeLLM(), reranker=FakeReranker())
    ingest(corpus, settings=settings, store=runtime.store, embedder=emb)
    with TestClient(create_app(runtime), base_url=BASE) as c:
        yield c
    runtime.close()


def test_index_page_and_security_headers(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200 and "local-rag" in res.text
    csp = res.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["referrer-policy"] == "no-referrer"
    assert "access-control-allow-origin" not in res.headers
    assert client.get("/static/app.js").status_code == 200


def test_no_docs_routes(client: TestClient) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_status_and_documents(client: TestClient) -> None:
    status = client.get("/api/status").json()
    assert status["documents"] == 4 and status["chat_model"] == "fake:llm"
    assert client.get("/api/status").headers["cache-control"] == "no-store"
    assert {d["source"] for d in client.get("/api/documents").json()} >= {"bread.txt"}


def test_ask_streams_ndjson(client: TestClient) -> None:
    res = client.post("/api/ask", json={"question": "How do I bake the loaf?", "mode": "hybrid"})
    assert res.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in res.text.splitlines()]
    assert events[0]["type"] == "sources" and events[-1]["type"] == "done"
    assert events[-1]["citations"] == [1]
    chunk_id = events[0]["sources"][0]["chunk_id"]
    chunk = client.get(f"/api/chunks/{chunk_id}").json()
    assert chunk["source"] == events[0]["sources"][0]["source"] and chunk["text"]
    assert client.get("/api/chunks/999999").status_code == 404


def test_ask_validation(client: TestClient) -> None:
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x", "mode": "evil"}).status_code == 422
    assert client.post("/api/ask", json={"question": "x", "top_k": 500}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 2001}).status_code == 422


def test_errors_are_streamed_not_raised(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> Iterator[dict[str, object]]:
        raise RuntimeError("model exploded")

    monkeypatch.setattr(RAG, "stream", boom)
    res = client.post("/api/ask", json={"question": "x"})
    assert json.loads(res.text.splitlines()[-1]) == {"type": "error", "message": "model exploded"}


def test_form_posts_are_refused(client: TestClient) -> None:
    res = client.post("/api/ask", data={"question": "x"})
    assert res.status_code == 415
    res = client.post(
        "/api/ask", content=b'{"question": "x"}', headers={"content-type": "text/plain"}
    )
    assert res.status_code == 415


def test_dns_rebinding_host_is_refused(client: TestClient) -> None:
    res = client.get("/api/status", headers={"host": "attacker.example:8765"})
    assert res.status_code == 400


@pytest.mark.parametrize(
    "origin",
    ["https://evil.example", "http://127.0.0.1:9999", "null", "http://localhost.evil:8765"],
)
def test_cross_origin_requests_are_refused(client: TestClient, origin: str) -> None:
    res = client.post("/api/ask", json={"question": "x"}, headers={"origin": origin})
    assert res.status_code == 403


def test_same_origin_is_allowed(client: TestClient) -> None:
    res = client.get("/api/status", headers={"origin": BASE})
    assert res.status_code == 200
