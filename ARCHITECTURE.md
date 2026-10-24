# Architecture

```
local_rag/
  config.py      Settings from LOCAL_RAG_* env vars; loopback checks
  loaders.py     discover() with path guards; Markdown / HTML / text / PDF -> Document(sections)
  chunking.py    structure-aware chunker -> Chunk(headings, page, text)
  embeddings.py  OllamaEmbedder, FastEmbedEmbedder (extra), HashingEmbedder
  ollama.py      tiny httpx client: /api/embed, /api/chat (stream), /api/tags
  store.py       SQLite: documents, chunks, FTS5 (BM25), sqlite-vec vec0 (cosine)
  ingest.py      discover -> load -> chunk -> embed -> store, incremental by sha256
  retrieval.py   bm25 | vector | hybrid (RRF) | hybrid+rerank; relevance gate
  prompting.py   untrusted-source framing, citation validation, refusal detection
  rag.py         retrieve -> prompt -> stream -> validate; ask() and stream()
  evaluation.py  bundled benchmark: recall@k, MRR, faithfulness judge, latency
  runtime.py     wires settings into store/embedder/retriever/RAG
  web.py         FastAPI app + static/ (index.html, app.js, style.css)
  cli.py         ingest | ask | chat | eval | serve | status
  data/          corpus/ (4 public-domain works) + qa.jsonl
```

## Data flow

1. **Discover.** Resolve the root, walk it without following directory symlinks, and
   keep regular files with a known suffix whose resolved path is inside the root.
2. **Load.** Each loader returns `Section(headings, text, page)`:
   - Markdown: ATX and setext headings outside code fences.
   - HTML: `h1`-`h6` via the stdlib `HTMLParser`; script and style are dropped.
   - Text: short ALL-CAPS or `CHAPTER`-style lines between blank lines.
   - PDF: one section per page (pypdf).
3. **Chunk.** Sections and pages are hard boundaries. Paragraphs are packed greedily up
   to `chunk_chars`. Over-long paragraphs split at sentences, then words. Each new chunk
   starts with up to `overlap_chars` of the previous one's tail, cut at a sentence or
   word boundary. The breadcrumb (`Title > Article 1 > Section 8`, or `(page 7)`) is
   prepended to the text that is embedded and BM25-indexed.
4. **Embed and store.** One transaction per document: a `documents` row, `chunks` rows
   (text plus the float32 vector), an FTS5 row and a `vec0` row with the same rowid.
   The index records its embedder name and dimension and refuses to mix them. Ingest
   skips files whose `sha256(bytes + embedder + chunk settings)` is unchanged.
5. **Retrieve.**
   - `bm25`: FTS5 `bm25()` over stemmed tokens. The query is reduced to quoted
     non-stopword tokens joined with `OR`.
   - `vector`: exact cosine KNN in `vec0`.
   - `hybrid`: reciprocal rank fusion `sum 1/(k + rank)` with `k = 60` over the top
     `candidates` of each list.
   - `hybrid+rerank`: a local cross-encoder re-scores the fused top `candidates`.
   - Relevance gate: when no query term occurs in the index *and* the best cosine is
     below `min_similarity`, we refuse without calling the model.
6. **Generate.** The system prompt sets the rules, including that sources are untrusted,
   that claims must be cited as `[n]`, and the exact refusal sentence. The user message
   holds the escaped `<sources>` block and then the question. Tokens are streamed. At
   the end, invented citations are stripped and refusals detected.

## Why sqlite-vec

| Option | Footprint | linux aarch64 + CI | Notes |
| --- | --- | --- | --- |
| **sqlite-vec** (chosen) | one ~150 KB loadable extension, no dependencies | manylinux aarch64 and x86_64 wheels | Lives in the same SQLite file as the chunks and the FTS5 BM25 index, so one file, one transaction, one backup. Exact KNN, which is plenty for personal corpora (10^5 chunks). No server, no telemetry. |
| Chroma | pulls onnxruntime, tokenizers, a client/server stack and an OpenTelemetry/PostHog telemetry layer you must switch off | wheels exist | Needs a separate keyword index for BM25 and has telemetry on by default. |
| LanceDB | pyarrow + the Rust lance engine (~100 MB) | wheels exist | Strong at scale and has full-text search, but heavyweight for a demo and a second storage format next to the metadata. |

If `enable_load_extension` is missing (some macOS system Pythons), the store falls back
to scanning the stored float32 blobs in Python. It gives the same results, only slower,
and a test checks that both paths agree.

## Embedders

- **ollama** (default): `nomic-embed-text` (768d, with `search_document:` and
  `search_query:` prefixes). `bge-m3`, `mxbai-embed-large`, `snowflake-arctic-embed`
  and others work by name.
- **fastembed** (`pip install 'local-rag-ollama[fastembed]'`): ONNX on CPU,
  `BAAI/bge-small-en-v1.5` (384d) by default. The weights are downloaded once into
  `~/.cache/local-rag`.
- **hashing**: signed feature hashing of stemmed unigrams and bigrams. No model and fully
  deterministic. A lexical baseline and the test embedder.

At query time the CLI and UI always use the embedder recorded in the index.

## Benchmark design

Gold labels are evidence *spans*, not chunk ids, so a chunking change doesn't silently
invalidate the QA set: a chunk is relevant when it contains the span from the right
document. `unmatched_evidence` lists any span that no chunk contains (a test keeps it
empty). Generation metrics need ollama:

- answer rate on answerable questions;
- refusal rate on unanswerable ones;
- citation rate and precision (cited chunks that are relevant);
- faithfulness: a local judge model rates each answer `supported`, `partial` or
  `unsupported` against the sources it was given, scored 1 / 0.5 / 0.

Each stage is timed: `embed_query`, `bm25`, `vector`, `fuse`, `rerank`, `generate`,
`first_token` and `judge`.
