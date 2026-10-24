# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md).

## Mission

`local-rag-ollama` answers questions about a folder of documents with cited answers,
entirely on the user's machine, and measures how well it does it.

## The rules

1. **Documents never leave the machine.** No hosted API, no telemetry, no new network
   call. `OLLAMA_HOST` stays loopback unless the user opts in.
2. **Retrieved text is untrusted.** It is escaped and delimited in prompts and rendered
   with `textContent` in the UI. Never interpolate it into HTML, SQL or a shell.
3. **Ingest stays inside the root.** Keep the symlink, hidden-file, size and `O_NOFOLLOW`
   guards in `loaders.py`. Every new loader gets a hostile test.
4. **Offline tests.** Hashing embedder, `FakeLLM`, `FakeReranker`, `httpx.MockTransport`.
   Never a real model, a download or the network.
5. **README numbers come from real runs.** Rerun `local-rag eval` and paste its table.
   Never edit a number by hand.
6. Every behaviour change gets a test, plus README and CHANGELOG entries.
7. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a test, or
   loosen lint/type/coverage config. Fix the code.

## Done means

`poetry run pytest` (coverage >= 85%), `ruff check`, `ruff format --check`,
`mypy` (strict), `bandit`, `pip-audit` all green; `python -m examples.quickstart` runs;
docs and CHANGELOG updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit secrets,
`.env` or index files (`*.sqlite3`).
