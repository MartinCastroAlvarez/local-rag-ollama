from __future__ import annotations

import io
import json
from functools import partial
from pathlib import Path

import pytest

from local_rag import cli
from local_rag.config import Settings
from local_rag.ollama import OllamaError
from local_rag.runtime import open_runtime
from tests.conftest import FakeJudge
from tests.conftest import FakeLLM
from tests.conftest import FakeReranker


def run(*argv: str, stdin: str = "") -> tuple[int, str]:
    out = io.StringIO()
    code = cli.main(list(argv), out=out, inp=io.StringIO(stdin))
    return code, out.getvalue()


@pytest.fixture
def db(tmp_path: Path) -> str:
    return str(tmp_path / "cli.sqlite3")


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(
        cli, "open_runtime", partial(open_runtime, llm=llm, reranker=FakeReranker())
    )
    return llm


def test_ingest_status_and_retrieve_only(corpus: Path, db: str) -> None:
    code, out = run("ingest", str(corpus), "--db", db, "--embedder", "hashing")
    assert code == 0 and "4 indexed" in out and "hashing:512" in out
    code, out = run("ingest", str(corpus), "--db", db, "--embedder", "hashing", "--prune")
    assert "4 unchanged" in out
    code, out = run("status", "--db", db)
    assert "documents: 4" in out and "guides/garden.md" in out
    # ask follows the index's embedder even when the default (ollama) is configured
    code, out = run("ask", "Saturn rings?", "--db", db, "--retrieve-only", "-k", "2")
    assert code == 0 and out.startswith("[1]") and "astronomy.pdf" in out


def test_ingest_reports_failures_and_rebuild(corpus: Path, db: str) -> None:
    (corpus / "bad.pdf").write_bytes(b"%PDF-1.4 nope")
    code, out = run("ingest", str(corpus), "--db", db, "--embedder", "hashing")
    assert code == 1 and "failed bad.pdf" in out
    code, out = run(
        "ingest", str(corpus / "bread.txt"), "--db", db, "--embedder", "hashing", "--rebuild"
    )
    assert "1 documents" in out


def test_embedder_mismatch_needs_rebuild(
    corpus: Path, db: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run("ingest", str(corpus), "--db", db, "--embedder", "hashing")
    monkeypatch.setenv("LOCAL_RAG_HASHING_DIM", "64")
    (corpus / "new.md").write_text("# New\n\nnew text")
    code, _ = run("ingest", str(corpus), "--db", db, "--embedder", "hashing")
    assert code == 2 and "--rebuild" in capsys.readouterr().err


def test_ask_streams_and_lists_sources(corpus: Path, db: str, fake_runtime: FakeLLM) -> None:
    run("ingest", str(corpus), "--db", db, "--embedder", "hashing")
    fake_runtime.replies = ["Pump road tyres to about 6 bar [1]."]
    code, out = run("ask", "What pressure for road tyres?", "--db", db)
    assert code == 0
    assert out.startswith("Pump road tyres") and "Sources:" in out and " *[1] bikes.html" in out


def test_ask_json(corpus: Path, db: str, fake_runtime: FakeLLM) -> None:
    run("ingest", str(corpus), "--db", db, "--embedder", "hashing")
    code, out = run("ask", "tomatoes sun", "--db", db, "--json", "--mode", "bm25")
    data = json.loads(out)
    assert code == 0 and data["citations"] == [1] and data["sources"][0]["n"] == 1
    assert data["refused"] is False and "retrieve" in data["timings"]


def test_chat_loop_keeps_history(corpus: Path, db: str, fake_runtime: FakeLLM) -> None:
    run("ingest", str(corpus), "--db", db, "--embedder", "hashing")
    code, out = run(
        "chat", "--db", db, stdin="How do I grow tomatoes?\n/reset\nand basil?\nmore?\n\n"
    )
    assert code == 0 and "history cleared" in out
    assert len(fake_runtime.calls) == 3
    assert len(fake_runtime.calls[1]) == 2  # after /reset: system + question
    assert len(fake_runtime.calls[2]) == 4  # one previous turn carried


def test_ask_without_ollama_fails_cleanly(
    corpus: Path, db: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run("ingest", str(corpus), "--db", db, "--embedder", "hashing")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens on loopback:9
    code, _ = run("ask", "tomatoes", "--db", db)
    assert code == 2 and "ollama serve" in capsys.readouterr().err


def test_eval_without_ollama_reports_retrieval_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_ollama(self: object) -> list[str]:
        raise OllamaError("down")

    monkeypatch.setattr("local_rag.ollama.OllamaClient.models", no_ollama)
    output = tmp_path / "eval.json"
    code, out = run(
        "eval", "--embedder", "hashing", "--modes", "bm25", "hybrid", "--output", str(output)
    )
    assert code == 0 and "generation metrics skipped" in out
    assert "| bm25 |" in out and "Run locally with ollama" in out
    result = json.loads(output.read_text())
    assert result["unmatched_evidence"] == [] and result["generation"] is None
    assert result["retrieval"]["bm25"]["recall@10"] > 0.9


def test_eval_with_fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_rag.ollama.OllamaClient.models", lambda self: ["qwen3:4b"])
    monkeypatch.setattr(
        cli,
        "OllamaChat",
        lambda client, model, **kw: FakeJudge() if kw.get("json_format") else FakeLLM(),
    )
    code, out = run("eval", "--embedder", "hashing", "--modes", "bm25")
    assert code == 0 and "Faithfulness" in out


def test_eval_with_missing_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_rag.ollama.OllamaClient.models", lambda self: [])
    code, out = run("eval", "--embedder", "hashing", "--modes", "bm25")
    assert code == 0 and "ollama pull" in out


def test_serve_refuses_non_loopback(capsys: pytest.CaptureFixture[str], db: str) -> None:
    code, _ = run("serve", "--host", "0.0.0.0", "--db", db)
    assert code == 2 and "loopback" in capsys.readouterr().err


def test_serve_runs_uvicorn_on_loopback(monkeypatch: pytest.MonkeyPatch, db: str) -> None:
    seen: dict[str, object] = {}

    def fake_run(app: object, **kwargs: object) -> None:
        seen.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    code, out = run("serve", "--db", db, "--port", "9999")
    assert code == 0 and seen["host"] == "127.0.0.1" and seen["port"] == 9999
    assert "http://127.0.0.1:9999/" in out


def test_bad_env_is_a_clean_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://198.51.100.7:11434")
    code, _ = run("status")
    assert code == 2 and "not loopback" in capsys.readouterr().err


def test_settings_from_args(tmp_path: Path) -> None:
    args = cli._parser().parse_args(
        ["ask", "q", "--mode", "bm25", "-k", "3", "--db", str(tmp_path / "x")]
    )
    settings = cli._settings(args)
    assert isinstance(settings, Settings) and settings.mode == "bm25" and settings.top_k == 3
