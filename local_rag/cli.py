"""``local-rag ingest | ask | chat | eval | serve | status``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import TextIO

from local_rag import __version__
from local_rag.config import EMBEDDERS
from local_rag.config import MODES
from local_rag.config import ConfigError
from local_rag.config import Settings
from local_rag.config import is_loopback_host
from local_rag.embeddings import make_embedder
from local_rag.ingest import ingest
from local_rag.loaders import IngestError
from local_rag.ollama import OllamaClient
from local_rag.ollama import OllamaError
from local_rag.rag import OllamaChat
from local_rag.runtime import Runtime
from local_rag.runtime import open_runtime
from local_rag.store import Store
from local_rag.store import StoreError


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, help="index file (env LOCAL_RAG_DB)")
    common.add_argument("--embedder", choices=EMBEDDERS, help="env LOCAL_RAG_EMBEDDER")
    common.add_argument("--embed-model", help="ollama embedding model (env LOCAL_RAG_EMBED_MODEL)")
    common.add_argument("--chat-model", help="ollama chat model (env LOCAL_RAG_CHAT_MODEL)")
    common.add_argument("-v", "--verbose", action="store_true")

    query = argparse.ArgumentParser(add_help=False)
    query.add_argument("--mode", choices=MODES, help="retrieval mode (env LOCAL_RAG_MODE)")
    query.add_argument("-k", "--top-k", type=int, help="chunks given to the model")

    parser = argparse.ArgumentParser(
        prog="local-rag", description="Fully local RAG over your documents with ollama."
    )
    parser.add_argument("--version", action="version", version=f"local-rag {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", parents=[common], help="index files or folders")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--rebuild", action="store_true", help="drop the index first")
    p.add_argument("--prune", action="store_true", help="forget files that are gone")

    p = sub.add_parser("ask", parents=[common, query], help="answer one question")
    p.add_argument("question")
    p.add_argument("--json", action="store_true", help="print the answer as JSON")
    p.add_argument("--retrieve-only", action="store_true", help="show chunks, skip the LLM")

    sub.add_parser("chat", parents=[common, query], help="interactive question loop")

    p = sub.add_parser("eval", parents=[common], help="run the bundled benchmark")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--generation-mode", choices=MODES, default="hybrid")
    p.add_argument("--no-generation", action="store_true", help="retrieval metrics only")
    p.add_argument("--judge-model", help="ollama judge model (env LOCAL_RAG_JUDGE_MODEL)")
    p.add_argument("--corpus", type=Path, help="folder to index instead of the bundled one")
    p.add_argument("--qa", type=Path, help="JSONL questions instead of the bundled set")
    p.add_argument("--output", type=Path, help="also write the full result as JSON")

    p = sub.add_parser("serve", parents=[common], help="web UI on 127.0.0.1")
    p.add_argument("--host", default="127.0.0.1", help="loopback address only")
    p.add_argument("--port", type=int, default=8765)

    sub.add_parser("status", parents=[common], help="show what is indexed")
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    return Settings.from_env().with_(
        db_path=args.db,
        embedder=args.embedder,
        embed_model=args.embed_model,
        chat_model=args.chat_model,
        mode=getattr(args, "mode", None),
        top_k=getattr(args, "top_k", None),
        judge_model=getattr(args, "judge_model", None),
    )


def _print_sources(sources: Sequence[dict[str, Any]], cited: Sequence[int], out: TextIO) -> None:
    if not sources:
        return
    out.write("\nSources:\n")
    for src in sources:
        mark = "*" if src["n"] in cited else " "
        out.write(f" {mark}[{src['n']}] {src['source']} - {src['location']}\n")


def cmd_ingest(args: argparse.Namespace, settings: Settings, out: TextIO) -> int:
    with Store(settings.db_path) as store:
        if args.rebuild:
            store.clear()
        embedder = make_embedder(settings)
        failed = 0
        for path in args.paths:
            report = ingest(
                path, settings=settings, store=store, embedder=embedder, prune=args.prune
            )
            out.write(f"{path}: {report.summary()}\n")
            for source, error in report.errors.items():
                out.write(f"  failed {source}: {error}\n")
            failed += len(report.errors)
        stats = store.stats()
        out.write(
            f"index {settings.db_path}: {stats['documents']} documents, {stats['chunks']} chunks, "
            f"{stats['embedder']} ({stats['vector_index']})\n"
        )
    return 1 if failed else 0


def _answer(
    runtime: Runtime,
    question: str,
    args: argparse.Namespace,
    out: TextIO,
    history: Sequence[tuple[str, str]] = (),
) -> str:
    """Stream one answer to ``out`` and return its final text."""
    text = ""
    sources: list[dict[str, Any]] = []
    cited: list[int] = []
    for event in runtime.rag.stream(question, mode=args.mode, top_k=args.top_k, history=history):
        if event["type"] == "sources":
            sources = event["sources"]
        elif event["type"] == "token":
            out.write(event["text"])
            out.flush()
        elif event["type"] == "done":
            text, cited = event["text"], event["citations"]
    out.write("\n")
    _print_sources(sources, cited, out)
    return text


def cmd_ask(args: argparse.Namespace, settings: Settings, out: TextIO) -> int:
    with open_runtime(settings) as runtime:
        if args.retrieve_only:
            retrieval = runtime.retriever.retrieve(args.question, mode=args.mode, top_k=args.top_k)
            for n, hit in enumerate(retrieval.hits, start=1):
                out.write(f"[{n}] {hit.score:.4f} {hit.chunk.source} - {hit.chunk.breadcrumb}\n")
                out.write(f"    {hit.chunk.text[:200]}\n")
            return 0
        if args.json:
            answer = runtime.rag.ask(args.question, mode=args.mode, top_k=args.top_k)
            json.dump(
                {
                    "question": answer.question,
                    "answer": answer.text,
                    "refused": answer.refused,
                    "citations": [c.number for c in answer.citations],
                    "sources": answer.sources,
                    "timings": answer.timings,
                },
                out,
                indent=2,
                ensure_ascii=False,
            )
            out.write("\n")
            return 0
        _answer(runtime, args.question, args, out)
    return 0


def cmd_chat(args: argparse.Namespace, settings: Settings, out: TextIO, inp: TextIO) -> int:
    history: list[tuple[str, str]] = []
    with open_runtime(settings) as runtime:
        out.write(
            f"local-rag chat ({runtime.store.stats()['chunks']} chunks). "
            "Empty line or /quit exits.\n"
        )
        while True:
            out.write("\n> ")
            out.flush()
            line = inp.readline()
            if not line or line.strip() in ("", "/quit", "/exit"):
                return 0
            if line.strip() == "/reset":
                history.clear()
                out.write("history cleared\n")
                continue
            text = _answer(runtime, line.strip(), args, out, history)
            history.append((line.strip(), text))


def cmd_eval(args: argparse.Namespace, settings: Settings, out: TextIO) -> int:
    from local_rag.evaluation import markdown_table
    from local_rag.evaluation import run_eval

    client = OllamaClient(settings.ollama_url, timeout_s=settings.timeout_s)
    llm = judge = None
    if not args.no_generation:
        try:
            available = client.models()
        except OllamaError as exc:
            out.write(f"ollama unavailable ({exc}); generation metrics skipped\n")
        else:
            missing = [m for m in {settings.chat_model, settings.judge_model} if m not in available]
            if missing:
                out.write(f"missing ollama models {missing}; run `ollama pull` first\n")
            else:
                llm = OllamaChat(client, settings.chat_model, temperature=settings.temperature)
                judge = OllamaChat(client, settings.judge_model, json_format=True)
    embedder = make_embedder(settings, client)
    try:
        result = run_eval(
            settings,
            embedder,
            modes=args.modes,
            llm=llm,
            judge=judge,
            corpus=args.corpus,
            qa_path=args.qa,
            generation_mode=args.generation_mode,
        )
    finally:
        client.close()
    corpus = result["corpus"]
    out.write(
        f"embedder {result['embedder']} ({result['vector_index']}), "
        f"{corpus['documents']} documents, "
        f"{corpus['chunks']} chunks, {result['questions']['answerable']} answerable + "
        f"{result['questions']['unanswerable']} unanswerable questions\n\n"
    )
    out.write(markdown_table(result) + "\n")
    if result["unmatched_evidence"]:
        out.write(f"\nwarning: no chunk contains the evidence of {result['unmatched_evidence']}\n")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings, out: TextIO) -> int:
    if not is_loopback_host(args.host):
        raise ConfigError(
            f"refusing to bind {args.host!r}: the web UI serves your documents and has no "
            "authentication, so it only listens on loopback (127.0.0.1, ::1, localhost)."
        )
    import uvicorn

    from local_rag.web import create_app

    runtime = open_runtime(settings)
    try:
        out.write(f"local-rag UI on http://{args.host}:{args.port}/ (Ctrl+C stops)\n")
        uvicorn.run(
            create_app(runtime),
            host=args.host,
            port=args.port,
            log_level="warning",
            server_header=False,
            proxy_headers=False,
        )
    finally:
        runtime.close()
    return 0


def cmd_status(args: argparse.Namespace, settings: Settings, out: TextIO) -> int:
    with Store(settings.db_path) as store:
        stats = store.stats()
        out.write(f"index {settings.db_path}\n")
        for key, value in stats.items():
            out.write(f"  {key}: {value}\n")
        for doc in store.documents():
            out.write(
                f"  - {doc['source']} ({doc['kind']}, {doc['chunks']} chunks): {doc['title']}\n"
            )
    return 0


def main(
    argv: Sequence[str] | None = None, *, out: TextIO | None = None, inp: TextIO | None = None
) -> int:
    args = _parser().parse_args(argv)
    out = out or sys.stdout
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    try:
        settings = _settings(args)
        if args.command == "ingest":
            return cmd_ingest(args, settings, out)
        if args.command == "ask":
            return cmd_ask(args, settings, out)
        if args.command == "chat":
            return cmd_chat(args, settings, out, inp or sys.stdin)
        if args.command == "eval":
            return cmd_eval(args, settings, out)
        if args.command == "serve":
            return cmd_serve(args, settings, out)
        return cmd_status(args, settings, out)
    except (ConfigError, IngestError, StoreError, OllamaError) as exc:
        sys.stderr.write(f"local-rag: {exc}\n")
        return 2
    except KeyboardInterrupt:  # pragma: no cover - interactive
        return 130
