"""A small local web UI: FastAPI + one static page, streaming NDJSON answers.

Hardening (the UI holds the full text of your documents):

* binds to loopback only (``serve`` refuses other hosts);
* ``Host`` must be ``127.0.0.1``/``localhost``/``::1`` (blocks DNS rebinding);
* a cross-origin ``Origin`` header gets 403, and no CORS headers are ever sent;
* ``POST /api/ask`` requires ``Content-Type: application/json`` (a plain HTML form from
  another site cannot send it without a CORS preflight, which is refused);
* a strict CSP (``script-src 'self'``, no inline script), ``nosniff``, ``no-referrer``,
  ``frame-ancestors 'none'``; the page renders document text with ``textContent`` only;
* no ingest endpoint: the browser can read the index, never point it at a new path;
* no OpenAPI/docs routes, no telemetry, no external assets.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterator
from importlib import resources
from typing import Any
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic import Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from local_rag.config import is_loopback_host
from local_rag.ollama import OllamaError
from local_rag.runtime import Runtime
from local_rag.store import StoreError

LOOPBACK_HOSTS = ["127.0.0.1", "localhost", "::1", "[::1]"]
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["bm25", "vector", "hybrid", "hybrid+rerank"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    history: list[tuple[str, str]] = Field(default_factory=list, max_length=6)


def _static_dir() -> str:
    return str(resources.files("local_rag") / "static")


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="local-rag", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOOPBACK_HOSTS)

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        origin = request.headers.get("origin")
        if origin is not None:
            parts = urlsplit(origin)
            host = request.headers.get("host", "")
            if not (is_loopback_host(parts.hostname or "") and parts.netloc == host):
                return JSONResponse({"error": "cross-origin request refused"}, status_code=403)
        if request.method not in ("GET", "HEAD"):
            content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type != "application/json":
                return JSONResponse(
                    {"error": "Content-Type must be application/json"}, status_code=415
                )
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(f"{_static_dir()}/index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=_static_dir()), name="static")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        settings = runtime.settings
        return {
            **runtime.store.stats(),
            "chat_model": runtime.rag.llm.name,
            "mode": settings.mode,
            "top_k": settings.top_k,
            "modes": ["bm25", "vector", "hybrid", "hybrid+rerank"],
        }

    @app.get("/api/documents")
    def documents() -> list[dict[str, object]]:
        return runtime.store.documents()

    @app.get("/api/chunks/{chunk_id}")
    def chunk(chunk_id: int) -> dict[str, Any]:
        found = runtime.store.get_chunks([chunk_id]).get(chunk_id)
        if found is None:
            raise HTTPException(status_code=404, detail="no such chunk")
        return {
            "chunk_id": found.id,
            "source": found.source,
            "location": found.breadcrumb,
            "page": found.page,
            "text": found.text,
        }

    @app.post("/api/ask")
    def ask(body: AskRequest) -> StreamingResponse:
        def events() -> Iterator[bytes]:
            try:
                for event in runtime.rag.stream(
                    body.question, mode=body.mode, top_k=body.top_k, history=body.history
                ):
                    yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
            except (OllamaError, StoreError, ValueError, RuntimeError) as exc:
                yield (json.dumps({"type": "error", "message": str(exc)[:500]}) + "\n").encode()

        return StreamingResponse(events(), media_type="application/x-ndjson")

    return app
