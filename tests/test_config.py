from __future__ import annotations

from pathlib import Path

import pytest

from local_rag.config import ConfigError
from local_rag.config import Settings
from local_rag.config import default_db_path
from local_rag.config import is_loopback_host


def test_defaults_are_local() -> None:
    s = Settings.from_env({})
    assert s.ollama_url == "http://127.0.0.1:11434"
    assert s.embedder == "ollama" and s.mode == "hybrid"
    assert s.db_path == default_db_path()


def test_env_overrides(tmp_path: Path) -> None:
    s = Settings.from_env(
        {
            "LOCAL_RAG_DB": str(tmp_path / "x.sqlite3"),
            "OLLAMA_HOST": "localhost:1234",
            "LOCAL_RAG_EMBEDDER": "hashing",
            "LOCAL_RAG_CHAT_MODEL": "llama3.2:3b",
            "LOCAL_RAG_MODE": "bm25",
            "LOCAL_RAG_TOP_K": "8",
            "LOCAL_RAG_MIN_SIMILARITY": "0.5",
            "LOCAL_RAG_THREADS": "2",
        }
    )
    assert s.db_path == tmp_path / "x.sqlite3"
    assert s.ollama_url == "http://localhost:1234"
    assert s.judge_model == "llama3.2:3b"  # judge follows the chat model unless set
    assert (s.embedder, s.mode, s.top_k, s.min_similarity, s.threads) == (
        "hashing",
        "bm25",
        8,
        0.5,
        2,
    )


@pytest.mark.parametrize(
    "env",
    [
        {"LOCAL_RAG_EMBEDDER": "openai"},
        {"LOCAL_RAG_MODE": "magic"},
        {"LOCAL_RAG_TOP_K": "zero"},
        {"LOCAL_RAG_TOP_K": "0"},
        {"LOCAL_RAG_MIN_SIMILARITY": "high"},
        {"LOCAL_RAG_CHUNK_CHARS": "300", "LOCAL_RAG_OVERLAP_CHARS": "300"},
    ],
)
def test_invalid_env(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_with_ignores_none_and_validates() -> None:
    s = Settings()
    assert s.with_(mode=None).mode == "hybrid"
    assert s.with_(mode="bm25").mode == "bm25"
    for bad in ({"mode": "x"}, {"embedder": "x"}, {"ollama_url": "http://10.0.0.1"}):
        with pytest.raises(ConfigError):
            s.with_(**bad)


@pytest.mark.parametrize(
    ("host", "ok"),
    [
        ("127.0.0.1", True),
        ("127.8.9.1", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("192.168.1.2", False),
        ("example.com", False),
    ],
)
def test_is_loopback(host: str, ok: bool) -> None:
    assert is_loopback_host(host) is ok


def test_xdg_data_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert default_db_path() == tmp_path / "local-rag" / "index.sqlite3"
