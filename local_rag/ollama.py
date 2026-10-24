"""A minimal ollama HTTP client: ``/api/embed``, ``/api/chat`` (streaming) and ``/api/tags``.

The client never follows redirects and ignores proxy environment variables, so the only
host that ever receives document text is the (loopback by default) ``OLLAMA_HOST``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from collections.abc import Sequence
from typing import Any

import httpx


class OllamaError(RuntimeError):
    """ollama is unreachable or answered with an error."""


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url
        self._http = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    def close(self) -> None:
        self._http.close()

    def _hint(self, model: str | None = None) -> str:
        pull = f" and `ollama pull {model}`" if model else ""
        return f"is ollama running at {self.base_url}? Try `ollama serve`{pull}."

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        model = payload.get("model")
        try:
            response = self._http.post(path, json=payload)
        except httpx.HTTPError as exc:
            raise OllamaError(f"{path}: {exc.__class__.__name__}; {self._hint(model)}") from exc
        if response.status_code != 200:
            raise OllamaError(f"{path}: HTTP {response.status_code} {_error(response)}")
        data = response.json()
        if not isinstance(data, dict):
            raise OllamaError(f"{path}: unexpected response")
        return data

    def models(self) -> list[str]:
        try:
            response = self._http.get("/api/tags")
        except httpx.HTTPError as exc:
            raise OllamaError(f"/api/tags: {exc.__class__.__name__}; {self._hint()}") from exc
        if response.status_code != 200:
            raise OllamaError(f"/api/tags: HTTP {response.status_code}")
        return [str(m.get("name", "")) for m in response.json().get("models", [])]

    def embed(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        data = self._post("/api/embed", {"model": model, "input": list(texts), "truncate": True})
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise OllamaError("/api/embed: wrong number of embeddings in the response")
        return [[float(x) for x in v] for v in vectors]

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        think: bool = False,
        json_format: bool = False,
    ) -> Iterator[str]:
        """Stream the assistant's content tokens for one chat turn."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": {"temperature": temperature},
        }
        if json_format:
            payload["format"] = "json"
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code != 200:
                    response.read()
                    raise OllamaError(f"/api/chat: HTTP {response.status_code} {_error(response)}")
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("error"):
                        raise OllamaError(f"/api/chat: {str(event['error'])[:300]}")
                    content = (event.get("message") or {}).get("content") or ""
                    if content:
                        yield content
                    if event.get("done"):
                        return
        except httpx.HTTPError as exc:
            raise OllamaError(f"/api/chat: {exc.__class__.__name__}; {self._hint(model)}") from exc


def _error(response: httpx.Response) -> str:
    try:
        return str(response.json().get("error", ""))[:300]
    except (ValueError, AttributeError):
        return response.text[:300]
