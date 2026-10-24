"""Settings, read from ``LOCAL_RAG_*`` environment variables (and CLI flags on top)."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

EMBEDDERS = ("ollama", "fastembed", "hashing")
MODES = ("bm25", "vector", "hybrid", "hybrid+rerank")

DEFAULT_REFUSAL = "I don't know based on the provided documents."


class ConfigError(ValueError):
    """A setting is invalid."""


def default_db_path() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "local-rag" / "index.sqlite3"


def is_loopback_host(host: str) -> bool:
    """True for ``localhost`` and loopback IP literals (``127.0.0.0/8``, ``::1``)."""
    host = host.strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_ollama_url(url: str, *, allow_remote: bool) -> str:
    """Return a normalised ollama base URL, refusing anything off this machine by default.

    Documents and questions go to this host, so "fully local" depends on it being loopback.
    """
    parts = urlsplit(url if "://" in url else f"http://{url}")
    if parts.scheme not in ("http", "https"):
        raise ConfigError(f"ollama URL must be http(s), got {parts.scheme!r}")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ConfigError("ollama URL must not carry credentials, a query or a fragment")
    host = parts.hostname or ""
    if not host:
        raise ConfigError("ollama URL has no host")
    if not allow_remote and not is_loopback_host(host):
        raise ConfigError(
            f"ollama host {host!r} is not loopback; documents would leave this machine. "
            "Set LOCAL_RAG_ALLOW_REMOTE_OLLAMA=1 if that host is yours."
        )
    netloc = parts.netloc
    if ":" not in netloc.rsplit("]", 1)[-1]:
        netloc = f"{netloc}:11434"
    return f"{parts.scheme}://{netloc}{parts.path.rstrip('/')}"


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _int(env: Mapping[str, str], key: str, default: int, *, minimum: int = 0) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    return value


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def _choice(env: Mapping[str, str], key: str, default: str, choices: tuple[str, ...]) -> str:
    value = env.get(key) or default
    if value not in choices:
        raise ConfigError(f"{key} must be one of {', '.join(choices)}; got {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    """Everything the pipeline needs. Immutable; derive variants with :meth:`with_`."""

    db_path: Path = field(default_factory=default_db_path)
    ollama_url: str = "http://127.0.0.1:11434"
    allow_remote_ollama: bool = False
    embedder: str = "ollama"
    embed_model: str = "nomic-embed-text"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    hashing_dim: int = 512
    chat_model: str = "qwen3:4b"
    judge_model: str = "qwen3:4b"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    mode: str = "hybrid"
    top_k: int = 5
    candidates: int = 30
    rrf_k: int = 60
    chunk_chars: int = 1200
    overlap_chars: int = 200
    min_similarity: float = 0.3
    max_file_bytes: int = 20 * 1024 * 1024
    threads: int = 0
    """CPU threads for fastembed's ONNX runtime; 0 lets onnxruntime decide."""
    temperature: float = 0.0
    timeout_s: float = 120.0
    refusal: str = DEFAULT_REFUSAL

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        allow_remote = _bool(env.get("LOCAL_RAG_ALLOW_REMOTE_OLLAMA"), False)
        base = cls()
        settings = cls(
            db_path=Path(env["LOCAL_RAG_DB"]).expanduser()
            if env.get("LOCAL_RAG_DB")
            else base.db_path,
            ollama_url=validate_ollama_url(
                env.get("OLLAMA_HOST") or base.ollama_url, allow_remote=allow_remote
            ),
            allow_remote_ollama=allow_remote,
            embedder=_choice(env, "LOCAL_RAG_EMBEDDER", base.embedder, EMBEDDERS),
            embed_model=env.get("LOCAL_RAG_EMBED_MODEL") or base.embed_model,
            fastembed_model=env.get("LOCAL_RAG_FASTEMBED_MODEL") or base.fastembed_model,
            hashing_dim=_int(env, "LOCAL_RAG_HASHING_DIM", base.hashing_dim, minimum=16),
            chat_model=env.get("LOCAL_RAG_CHAT_MODEL") or base.chat_model,
            judge_model=env.get("LOCAL_RAG_JUDGE_MODEL")
            or env.get("LOCAL_RAG_CHAT_MODEL")
            or base.judge_model,
            rerank_model=env.get("LOCAL_RAG_RERANK_MODEL") or base.rerank_model,
            mode=_choice(env, "LOCAL_RAG_MODE", base.mode, MODES),
            top_k=_int(env, "LOCAL_RAG_TOP_K", base.top_k, minimum=1),
            candidates=_int(env, "LOCAL_RAG_CANDIDATES", base.candidates, minimum=1),
            rrf_k=_int(env, "LOCAL_RAG_RRF_K", base.rrf_k, minimum=1),
            chunk_chars=_int(env, "LOCAL_RAG_CHUNK_CHARS", base.chunk_chars, minimum=200),
            overlap_chars=_int(env, "LOCAL_RAG_OVERLAP_CHARS", base.overlap_chars),
            min_similarity=_float(env, "LOCAL_RAG_MIN_SIMILARITY", base.min_similarity),
            max_file_bytes=_int(env, "LOCAL_RAG_MAX_FILE_BYTES", base.max_file_bytes, minimum=1),
            threads=_int(env, "LOCAL_RAG_THREADS", base.threads),
            temperature=_float(env, "LOCAL_RAG_TEMPERATURE", base.temperature),
            timeout_s=_float(env, "LOCAL_RAG_TIMEOUT", base.timeout_s),
        )
        settings.check()
        return settings

    def check(self) -> None:
        if self.embedder not in EMBEDDERS:
            raise ConfigError(f"unknown embedder {self.embedder!r}")
        if self.mode not in MODES:
            raise ConfigError(f"unknown mode {self.mode!r}")
        if self.overlap_chars >= self.chunk_chars:
            raise ConfigError("overlap must be smaller than the chunk size")
        validate_ollama_url(self.ollama_url, allow_remote=self.allow_remote_ollama)

    def with_(self, **changes: Any) -> Settings:
        """Return a copy with ``changes`` applied (``None`` values are ignored)."""
        updated = replace(self, **{k: v for k, v in changes.items() if v is not None})
        updated.check()
        return updated
