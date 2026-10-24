# Contributing to local-rag-ollama

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install                       # add -E fastembed for the ONNX embedder + reranker
poetry run pytest
poetry run ruff check local_rag tests examples scripts && poetry run ruff format --check local_rag tests examples scripts
poetry run mypy local_rag
poetry run bandit -c pyproject.toml -q -r local_rag
poetry run pip-audit
poetry run python -m examples.quickstart
```

Open a PR against `main` from a `feat/...`, `fix/...`, `docs/...` or `chore/...` branch.

## What this project will not accept

- Tests that call a real model, download weights or touch the network.
- Hosted LLM or embedding providers as defaults, telemetry, or update checks.
- Rendering document or model text as HTML in the UI.
- Ingest code that follows symlinks out of the root or reads hidden files.
- Hand-edited benchmark numbers.

## What it does accept

- New loaders (with a hostile-input test), better chunking, new retrieval modes.
- More questions in `local_rag/data/qa.jsonl`. Each needs a verbatim `evidence` span,
  and `test_every_evidence_span_survives_default_chunking` must stay green.
- New public-domain corpus texts (with their licence checked and recorded in
  `local_rag/data/README.md`).

## Updating the benchmark table

```bash
poetry install -E fastembed
local-rag eval --embedder fastembed --output eval.json        # + ollama for generation
local-rag eval --embedder hashing
```

Paste the printed tables into the README together with the embedder, machine and date.

## House rules

1. Tests alongside features; security-relevant changes get a hostile case.
2. README, ARCHITECTURE/SECURITY (when relevant) and CHANGELOG change with the code.
3. No secrets in commits; `.env.example` holds defaults only.
