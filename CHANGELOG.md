# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-24

### Added
- Ingest of Markdown, plain text, HTML and PDF (pypdf) folders, incremental by content
  hash, with `--rebuild` and `--prune`.
- Structure-aware chunking: sections and PDF pages are hard boundaries, paragraphs are
  packed to `LOCAL_RAG_CHUNK_CHARS` with sentence-boundary overlap, and the heading
  breadcrumb is indexed with each chunk.
- One SQLite index: FTS5 (BM25, porter stemming) plus a sqlite-vec `vec0` cosine index,
  with a pure-Python vector scan fallback when extensions can't be loaded.
- Embedders: ollama (`nomic-embed-text` by default, task prefixes for nomic, mxbai and
  snowflake), fastembed ONNX (optional extra), and a dependency-free hashing embedder.
- Retrieval modes `bm25`, `vector`, `hybrid` (reciprocal rank fusion) and
  `hybrid+rerank` (local fastembed cross-encoder), plus a relevance gate.
- Answers from any ollama chat model, streamed, with `[n]` citations validated against
  the sources and a fixed refusal sentence when the context is insufficient.
- CLI: `local-rag ingest | ask | chat | eval | serve | status`.
- Web UI (FastAPI + one static page): streaming answers, clickable citations, source
  viewer.
- Benchmark: a bundled public-domain corpus (four works, four formats) and 45 questions;
  recall@k, MRR@10, answer/refusal/citation rates, judge-graded faithfulness, and
  per-stage latency.
- Security: delimited and escaped untrusted sources, citation validation, ingest
  path-traversal and symlink guards, loopback-only ollama and UI, Host/Origin checks, a
  strict CSP, no telemetry.
